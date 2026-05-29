
# backend/api/inference.py
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.auth.auth_service import require_login

require_login()

PROTOCOLS = ["Contraction", "Cough", "Defecation", "RestPressure", "rair"]
PROTOCOL_ALIASES = {
    "contraction": "Contraction",
    "cough": "Cough",
    "defecation": "Defecation",
    "restpressure": "RestPressure",
    "rest_pressure": "RestPressure",
    "rest": "RestPressure",
    "rair": "rair",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

ARTIFACT_DIR = ROOT_DIR / "outputs" / "inference_artifacts"
SCALER_MEAN_PATH = ARTIFACT_DIR / "scaler_mean.npy"
SCALER_SCALE_PATH = ARTIFACT_DIR / "scaler_scale.npy"
PCA_COMPONENTS_PATH = ARTIFACT_DIR / "pca_components.npy"
PCA_MEAN_PATH = ARTIFACT_DIR / "pca_mean.npy"
CLUSTER_PROTOTYPES_PATH = ARTIFACT_DIR / "cluster_prototypes_pca.npy"
SIMILAR_CASES_PATH = ARTIFACT_DIR / "reference_patient_embeddings_pca.csv"

MODEL_NAME = "dinov2_vitb14"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DINO_MODEL = None


def _normalize_protocol_name(name: str) -> str:
    if name in PROTOCOLS:
        return name
    return PROTOCOL_ALIASES.get(str(name).strip(), PROTOCOL_ALIASES.get(str(name).strip().lower(), str(name).strip()))


def normalize_protocol_files(protocol_files: Dict[str, List]) -> Dict[str, List]:
    normalized = {p: [] for p in PROTOCOLS}
    if not isinstance(protocol_files, dict):
        return normalized
    for key, files in protocol_files.items():
        canonical = _normalize_protocol_name(key)
        if canonical not in normalized:
            continue
        normalized[canonical].extend(files or [])
    return normalized


def maybe_imagenet_normalize(arr: np.ndarray) -> np.ndarray:
    amax = float(arr.max())
    amin = float(arr.min())

    if amin < -0.1:
        return arr

    if amax > 1.5:
        arr = arr / 255.0

    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def validate_inference_input(payload: dict, protocol_files: Dict[str, List]) -> List[str]:
    errors = []

    patient_id = str((payload or {}).get("patient_id", "")).strip()
    if not patient_id:
        errors.append("患者ID不能为空。")

    protocol_files = normalize_protocol_files(protocol_files)
    total_files = sum(len(v) for v in protocol_files.values())
    if total_files == 0:
        errors.append("请至少上传一张图片。")

    has_any_core = any(
        (payload or {}).get(k) is not None
        for k in ["resting_pressure", "msp", "defecatory_rectal_pressure"]
    )
    if not has_any_core:
        errors.append("核心临床指标至少填写 1 项：静息压、MSP、排便时直肠压力。")

    return errors


def pil_to_npy_rgb224(file_obj) -> np.ndarray:
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass
    image = Image.open(file_obj).convert("RGB").resize((224, 224))
    arr = np.asarray(image).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = maybe_imagenet_normalize(arr)
    return arr.astype("float32")


def load_dinov2(model_name: str, device: torch.device):
    try:
        base = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:
        raise RuntimeError(
            f"torch.hub 加载 {model_name} 失败。"
            f"常见原因：环境不能联网 / torch.hub 被限制。"
            f"原始错误：{repr(e)}"
        )

    # FP16 推理：Jetson Orin 有良好 FP16 支持，可减少约 50% 显存
    use_fp16 = device.type == "cuda"
    if use_fp16:
        base.half()
    base.eval().to(device)

    class DinoWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        @torch.inference_mode()
        def forward(self, x):
            out = self.m(x)

            if isinstance(out, (list, tuple)):
                out = out[0]

            if isinstance(out, dict):
                for k in ["x_norm_clstoken", "x", "feats", "features"]:
                    if k in out:
                        out = out[k]
                        break
                else:
                    out = next(iter(out.values()))

            if getattr(out, "ndim", 0) == 3:
                out = out[:, 0, :]

            return out

    return DinoWrapper(base)


def get_dino_model():
    global _DINO_MODEL
    if _DINO_MODEL is None:
        _DINO_MODEL = load_dinov2(MODEL_NAME, DEVICE)
    return _DINO_MODEL


def extract_embeddings_from_uploaded_files(protocol_files: Dict[str, List]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    model = get_dino_model()
    protocol_files = normalize_protocol_files(protocol_files)

    xs = []
    meta = []

    for proto in PROTOCOLS:
        files = protocol_files.get(proto, [])
        for f in files:
            arr = pil_to_npy_rgb224(f)
            xs.append(arr)
            meta.append((proto, getattr(f, "name", "uploaded_image")))

    if len(xs) == 0:
        raise ValueError("未检测到可用于推理的图片。")

    x = torch.from_numpy(np.stack(xs, axis=0)).to(DEVICE)
    # FP16 推理：与模型精度一致
    if DEVICE.type == "cuda":
        x = x.half()

    with torch.inference_mode():
        y = model(x).detach().cpu().float().numpy().astype("float32")

    # 释放 GPU 临时显存（Jetson 8GB 统一内存需要及时回收）
    if DEVICE.type == "cuda":
        del x
        torch.cuda.empty_cache()

    return y, meta


def aggregate_patient_vector_attention(
    embeddings: np.ndarray,
    meta: List[Tuple[str, str]],
    protocols: List[str],
    temperature: float = 0.07,
    topk: int = 8,
) -> Tuple[np.ndarray, List[dict]]:
    D = embeddings.shape[1]
    bucket = {p: [] for p in protocols}
    file_bucket = {p: [] for p in protocols}

    for emb, (proto, fname) in zip(embeddings, meta):
        bucket.setdefault(proto, []).append(emb.astype("float32"))
        file_bucket.setdefault(proto, []).append(fname)

    proto_vecs = []
    details = []

    for proto in protocols:
        items = bucket.get(proto, [])
        fnames = file_bucket.get(proto, [])

        if len(items) == 0:
            proto_vecs.append(np.zeros((D,), dtype="float32"))
            continue

        E = np.stack(items, axis=0)
        En = _l2norm(E, axis=1)
        centroid = _l2norm(En.mean(axis=0, keepdims=True)).squeeze(0)
        scores = (En @ centroid).astype("float32")

        if topk > 0 and len(scores) > topk:
            idx = np.argsort(-scores)[:topk]
            E = E[idx]
            scores = scores[idx]
            fnames = [fnames[i] for i in idx]

        w = _softmax((scores / float(temperature)).reshape(1, -1), axis=1).reshape(-1)
        z = (w[:, None] * E).sum(axis=0).astype("float32")
        proto_vecs.append(z)

        ranked_idx = np.argsort(-scores)
        for rank_pos, i in enumerate(ranked_idx, start=1):
            details.append({
                "protocol": proto,
                "rank": rank_pos,
                "filename": fnames[i],
                "score": float(scores[i]),
                "weight": float(w[i]),
            })

    patient_vec = np.concatenate(proto_vecs, axis=0).astype("float32")
    return patient_vec, details


def load_inference_artifacts():
    required = [
        SCALER_MEAN_PATH,
        SCALER_SCALE_PATH,
        PCA_COMPONENTS_PATH,
        PCA_MEAN_PATH,
        CLUSTER_PROTOTYPES_PATH,
        SIMILAR_CASES_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少在线推理所需离线产物，请先生成以下文件：\n" + "\n".join(missing)
        )

    return {
        "scaler_mean": np.load(SCALER_MEAN_PATH),
        "scaler_scale": np.load(SCALER_SCALE_PATH),
        "pca_components": np.load(PCA_COMPONENTS_PATH),
        "pca_mean": np.load(PCA_MEAN_PATH),
        "cluster_prototypes": np.load(CLUSTER_PROTOTYPES_PATH),
        "ref_df": pd.read_csv(SIMILAR_CASES_PATH),
    }


def transform_patient_vector(patient_vec: np.ndarray, artifacts: dict) -> np.ndarray:
    x = patient_vec.reshape(1, -1)
    x_scaled = (x - artifacts["scaler_mean"]) / (artifacts["scaler_scale"] + 1e-12)
    x_pca = (x_scaled - artifacts["pca_mean"]) @ artifacts["pca_components"].T
    return x_pca.reshape(-1).astype("float32")


def assign_cluster_and_confidence(patient_vec_pca: np.ndarray, artifacts: dict):
    prototypes = artifacts["cluster_prototypes"]
    sims = cosine_similarity(patient_vec_pca.reshape(1, -1), prototypes).reshape(-1)

    pred_cluster = int(np.argmax(sims))
    conf = float((sims[pred_cluster] + 1.0) / 2.0)
    conf = max(0.0, min(1.0, conf))
    is_boundary = conf < 0.8

    return pred_cluster, conf, is_boundary, sims


def find_similar_cases(patient_vec_pca: np.ndarray, pred_cluster: int, artifacts: dict, topk: int = 3):
    ref_df = artifacts["ref_df"].copy()
    feat_cols = [c for c in ref_df.columns if c.startswith("pc")]
    required_cols = {"patient_id", "consensus_cluster"}
    if not feat_cols or not required_cols.issubset(ref_df.columns):
        return []

    ref_df = ref_df[ref_df["consensus_cluster"] == pred_cluster].copy()
    if ref_df.empty:
        return []

    Xref = ref_df[feat_cols].values.astype("float32")
    sims = cosine_similarity(patient_vec_pca.reshape(1, -1), Xref).reshape(-1)
    ref_df["sim"] = sims

    top = ref_df.sort_values("sim", ascending=False).head(topk)
    return top["patient_id"].astype(str).tolist()


def build_summary(pred_cluster: int, confidence: float, payload: dict) -> str:
    rp = payload.get("resting_pressure")
    msp = payload.get("msp")
    drp = payload.get("defecatory_rectal_pressure")

    parts = [f"该患者更接近 Cluster {pred_cluster}。"]

    hints = []
    if rp is not None and rp < 40:
        hints.append("静息压偏低")
    if msp is not None and msp < 100:
        hints.append("MSP 偏低")
    if drp is not None and drp < 45:
        hints.append("排便时直肠推进压力不足")

    if hints:
        parts.append("当前输入提示：" + "、".join(hints) + "。")

    parts.append(f"模型映射置信度约为 {confidence:.2%}。")
    if confidence < 0.8:
        parts.append("该结果属于边界型，建议结合原始图像和临床报告人工复核。")

    return "".join(parts)


def run_real_inference(payload: dict, protocol_files: Dict[str, List]) -> dict:
    protocol_files = normalize_protocol_files(protocol_files)
    embeddings, meta = extract_embeddings_from_uploaded_files(protocol_files)

    patient_vec, attn_details = aggregate_patient_vector_attention(
        embeddings=embeddings,
        meta=meta,
        protocols=PROTOCOLS,
        temperature=0.07,
        topk=8,
    )

    artifacts = load_inference_artifacts()
    patient_vec_pca = transform_patient_vector(patient_vec, artifacts)

    pred_cluster, confidence, is_boundary, _ = assign_cluster_and_confidence(
        patient_vec_pca, artifacts
    )

    similar_cases = find_similar_cases(
        patient_vec_pca=patient_vec_pca,
        pred_cluster=pred_cluster,
        artifacts=artifacts,
        topk=3,
    )

    summary = build_summary(pred_cluster, confidence, payload)

    return {
        "predicted_cluster": pred_cluster,
        "confidence": round(float(confidence), 4),
        "is_boundary": bool(is_boundary),
        "similar_cases": similar_cases,
        "summary": summary,
        "model_version": "dinov2_attn_pooling_online_v2",
        "inference_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_attention_details": attn_details,
        "protocols_used": [p for p in PROTOCOLS if len(protocol_files.get(p, [])) > 0],
    }


def run_inference(payload: dict, protocol_files: Dict[str, List]) -> dict:
    payload = payload or {}
    protocol_files = normalize_protocol_files(protocol_files)
    errors = validate_inference_input(payload, protocol_files)
    if errors:
        return {"ok": False, "errors": errors}

    try:
        result = run_real_inference(payload, protocol_files)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "errors": [f"推理失败：{e}"]}

# sctripts/M6_AttnVLM_score_reweight_from_cache.py
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


PROJECT_DIR = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")

CACHE_DIR = Path(r"H:\windows\图像数据\dataProcess\outputs\cache")
IMAGE_EMB_PATH = CACHE_DIR / "image_embeddings.npy"
IMAGE_META_PATH = CACHE_DIR / "image_meta.csv"

# 当前你已经有 mock VLM 分数和 reweight factor
VLM_SCORE_CSV = PROJECT_DIR / "outputs" / "vlm" / "vlm_image_scores_m1_topk4_mock.csv"

OUT_DIR = PROJECT_DIR / "outputs" /  "m6_attn_vlm_topk4_mock_from_cache_k3"

PROTOCOL_ORDER = [
    "RestPressure",
    "Contraction",
    "Defecation",
    "Cough",
    "rair",
]

N_PCA = 50
N_CLUSTERS = 3
RANDOM_STATE = 42


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"无法读取 CSV: {path}")


def norm_patient_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def norm_protocol(x):
    s = str(x).strip()
    sl = s.lower()

    mapping = {
        "restpressure": "RestPressure",
        "rest pressure": "RestPressure",
        "rest_pressure": "RestPressure",
        "contraction": "Contraction",
        "cough": "Cough",
        "defecation": "Defecation",
        "rair": "rair",
    }
    return mapping.get(sl, s)


def basename_key(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\\", "/")
    return Path(s).name.lower()


def make_key(df: pd.DataFrame, patient_col: str, protocol_col: str, path_col: str) -> pd.Series:
    return (
        df[patient_col].map(norm_patient_id)
        + "|"
        + df[protocol_col].map(norm_protocol)
        + "|"
        + df[path_col].map(basename_key)
    )


def choose_vlm_path_col(df: pd.DataFrame) -> str:
    candidates = [
        "image_path",
        "vlm_input_path",
        "feature_path_resolved",
        "feature_path_raw",
        "filepath",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到图像路径列。当前列名: {list(df.columns)}")


def ensure_numeric(s, name):
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().any():
        n_bad = int(out.isna().sum())
        print(f"[警告] {name} 有 {n_bad} 个非数值，已填充为 0")
        out = out.fillna(0.0)
    return out.astype(float)


def normalize_weights_by_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["final_weight_norm"] = 0.0

    for (pid, proto), idx in df.groupby(["patient_id_norm", "protocol_norm"]).groups.items():
        idx = list(idx)
        raw = df.loc[idx, "final_weight_raw"].to_numpy(dtype=float)
        attn = df.loc[idx, "attention_weight"].to_numpy(dtype=float)

        raw = np.clip(raw, 0.0, None)
        raw_sum = raw.sum()

        if raw_sum > 0:
            w = raw / raw_sum
        else:
            attn = np.clip(attn, 0.0, None)
            attn_sum = attn.sum()
            if attn_sum > 0:
                w = attn / attn_sum
            else:
                w = np.ones(len(idx), dtype=float) / max(len(idx), 1)

        df.loc[idx, "final_weight_norm"] = w

    return df


def natural_patient_sort(ids):
    def key_func(x):
        s = str(x)
        if re.fullmatch(r"\d+", s):
            return (0, int(s))
        return (1, s)
    return sorted(ids, key=key_func)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] 加载 cache embedding 与 meta")
    image_emb = np.load(IMAGE_EMB_PATH, mmap_mode="r")
    image_meta = read_csv_smart(IMAGE_META_PATH)

    print("image_embeddings:", image_emb.shape, image_emb.dtype)
    print("image_meta:", image_meta.shape)
    print("image_meta columns:", list(image_meta.columns))

    assert len(image_meta) == image_emb.shape[0], (
        f"image_meta 行数 {len(image_meta)} != image_embeddings 第一维 {image_emb.shape[0]}"
    )

    print("\n[2] 加载 M1 topk4 VLM score CSV")
    scores = read_csv_smart(VLM_SCORE_CSV)
    print("scores:", scores.shape)
    print("scores columns:", list(scores.columns))

    required_cols = ["patient_id", "protocol", "attention_weight"]
    for c in required_cols:
        if c not in scores.columns:
            raise ValueError(f"缺少必要列: {c}")

    vlm_path_col = choose_vlm_path_col(scores)
    print("使用图像路径列:", vlm_path_col)

    print("\n[3] 构造 cache ↔ M1 topk4 对齐索引")
    meta_key = make_key(image_meta, "patient_id", "protocol", "filepath")

    duplicated_meta_keys = int(meta_key.duplicated().sum())
    if duplicated_meta_keys > 0:
        print(f"[警告] image_meta 中存在重复 key: {duplicated_meta_keys}")

    key_to_index = {}
    for i, k in enumerate(meta_key):
        if k not in key_to_index:
            key_to_index[k] = i

    score_key = make_key(scores, "patient_id", "protocol", vlm_path_col)
    matched_indices = np.array([key_to_index.get(k, -1) for k in score_key], dtype=np.int64)

    n_match = int((matched_indices >= 0).sum())
    n_total = len(scores)
    print(f"匹配成功: {n_match}/{n_total} = {n_match / max(n_total, 1):.4f}")

    if n_match != n_total:
        missing = scores.loc[matched_indices < 0, ["patient_id", "protocol", vlm_path_col]].head(20)
        missing.to_csv(OUT_DIR / "missing_alignment_examples.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            f"存在未匹配图像: {n_total - n_match}。示例已保存到 missing_alignment_examples.csv"
        )

    print("\n[4] 取出 M1 topk4 对应 frame embeddings")
    X_frames = np.asarray(image_emb[matched_indices], dtype=np.float32)
    print("selected frame embeddings:", X_frames.shape)

    scores = scores.copy()
    scores["patient_id_norm"] = scores["patient_id"].map(norm_patient_id)
    scores["protocol_norm"] = scores["protocol"].map(norm_protocol)
    scores["cache_index"] = matched_indices

    scores["attention_weight"] = ensure_numeric(scores["attention_weight"], "attention_weight")

    if "vlm_reweight_factor" in scores.columns:
        scores["vlm_reweight_factor"] = ensure_numeric(scores["vlm_reweight_factor"], "vlm_reweight_factor")
    else:
        print("[提示] 没有 vlm_reweight_factor，全部置为 1.0")
        scores["vlm_reweight_factor"] = 1.0

    scores["final_weight_raw"] = scores["attention_weight"] * scores["vlm_reweight_factor"]
    scores = normalize_weights_by_group(scores)

    print("\n[5] 协议级 weighted pooling")
    scores["_frame_row"] = np.arange(len(scores), dtype=np.int64)

    protocol_embeddings = {}
    group_rows = []

    for (pid, proto), g in scores.groupby(["patient_id_norm", "protocol_norm"], sort=False):
        rows = g["_frame_row"].to_numpy(dtype=np.int64)
        w = g["final_weight_norm"].to_numpy(dtype=np.float32)

        emb = (X_frames[rows] * w[:, None]).sum(axis=0).astype(np.float32)
        protocol_embeddings[(pid, proto)] = emb

        group_rows.append({
            "patient_id": pid,
            "protocol": proto,
            "n_frames": int(len(g)),
            "attention_sum": float(g["attention_weight"].sum()),
            "vlm_factor_mean": float(g["vlm_reweight_factor"].mean()),
            "vlm_factor_min": float(g["vlm_reweight_factor"].min()),
            "vlm_factor_max": float(g["vlm_reweight_factor"].max()),
            "final_weight_raw_sum": float(g["final_weight_raw"].sum()),
            "final_weight_norm_sum": float(g["final_weight_norm"].sum()),
        })

    protocol_stats = pd.DataFrame(group_rows)
    protocol_stats.to_csv(OUT_DIR / "protocol_pooling_stats.csv", index=False, encoding="utf-8-sig")

    print("协议级 embedding 数:", len(protocol_embeddings))

    print("\n[6] 患者级 concat embedding")
    patient_ids = natural_patient_sort(scores["patient_id_norm"].drop_duplicates().tolist())

    dim = X_frames.shape[1]
    zero = np.zeros(dim, dtype=np.float32)

    X_patient = []
    presence_rows = []

    for pid in patient_ids:
        vecs = []
        presence = {"patient_id": pid}

        for proto in PROTOCOL_ORDER:
            key = (pid, proto)
            has_proto = key in protocol_embeddings
            presence[f"has_{proto}"] = int(has_proto)
            vecs.append(protocol_embeddings.get(key, zero))

        X_patient.append(np.concatenate(vecs, axis=0))
        presence_rows.append(presence)

    X_patient = np.stack(X_patient).astype(np.float32)
    presence_df = pd.DataFrame(presence_rows)
    presence_df["protocol_present_count"] = presence_df[[f"has_{p}" for p in PROTOCOL_ORDER]].sum(axis=1)

    print("patient_embeddings_concat:", X_patient.shape)
    print("患者数:", len(patient_ids))
    print("协议顺序:", PROTOCOL_ORDER)

    np.save(OUT_DIR / "patient_embeddings_concat.npy", X_patient)

    with open(OUT_DIR / "patient_ids.txt", "w", encoding="utf-8") as f:
        for pid in patient_ids:
            f.write(str(pid) + "\n")

    presence_df.to_csv(OUT_DIR / "patient_protocol_presence.csv", index=False, encoding="utf-8-sig")

    print("\n[7] PCA")
    n_pca = min(N_PCA, X_patient.shape[0] - 1, X_patient.shape[1])
    if n_pca < 2:
        raise RuntimeError(f"PCA 维度过小: n_pca={n_pca}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_patient)

    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    print("patient_embeddings_pca:", X_pca.shape)
    print("PCA explained variance ratio sum:", float(pca.explained_variance_ratio_.sum()))

    np.save(OUT_DIR / "patient_embeddings_pca.npy", X_pca)
    np.save(OUT_DIR / "pca_mean.npy", pca.mean_.astype(np.float32))
    np.save(OUT_DIR / "pca_components.npy", pca.components_.astype(np.float32))
    np.save(OUT_DIR / "scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(OUT_DIR / "scaler_scale.npy", scaler.scale_.astype(np.float32))

    with open(OUT_DIR / "pca_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_pca": int(n_pca),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }, f, ensure_ascii=False, indent=2)

    print("\n[8] KMeans clustering")
    if len(patient_ids) > N_CLUSTERS and N_CLUSTERS >= 2:
        km = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=50,
        )
        labels = km.fit_predict(X_pca)

        metrics = {
            "method": "KMeans",
            "n_clusters": int(N_CLUSTERS),
            "n_patients": int(len(patient_ids)),
            "input": "patient_embeddings_pca",
            "silhouette_score": float(silhouette_score(X_pca, labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(X_pca, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(X_pca, labels)),
            "inertia": float(km.inertia_),
            "cluster_counts": {
                str(k): int(v)
                for k, v in pd.Series(labels).value_counts().sort_index().items()
            },
        }
    else:
        labels = np.full(len(patient_ids), -1, dtype=int)
        metrics = {
            "method": "KMeans",
            "error": "样本数不足，未执行聚类",
            "n_clusters": int(N_CLUSTERS),
            "n_patients": int(len(patient_ids)),
        }

    clusters = pd.DataFrame({
        "patient_id": patient_ids,
        "cluster": labels,
    })

    clusters = clusters.merge(
        presence_df[["patient_id", "protocol_present_count"]],
        on="patient_id",
        how="left",
    )

    for i in range(min(5, X_pca.shape[1])):
        clusters[f"PC{i + 1}"] = X_pca[:, i]

    clusters.to_csv(OUT_DIR / "clusters.csv", index=False, encoding="utf-8-sig")

    with open(OUT_DIR / "cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("cluster metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n[9] 保存对齐与权重明细")
    keep_cols = [
        "patient_id",
        "protocol",
        "rank",
        vlm_path_col,
        "cache_index",
        "attention_weight",
        "vlm_reweight_factor",
        "final_weight_raw",
        "final_weight_norm",
    ]

    optional_cols = [
        "centroid_score",
        "vlm_score_raw",
        "vlm_score_norm",
        "vlm_image_quality",
        "vlm_pattern_label",
        "vlm_uncertain",
        "vlm_mode",
        "vlm_reason",
    ]

    keep_cols += [c for c in optional_cols if c in scores.columns]
    scores[keep_cols].to_csv(OUT_DIR / "topk_frame_scores_aligned.csv", index=False, encoding="utf-8-sig")

    alignment_report = {
        "image_embeddings_path": str(IMAGE_EMB_PATH),
        "image_meta_path": str(IMAGE_META_PATH),
        "vlm_score_csv": str(VLM_SCORE_CSV),
        "out_dir": str(OUT_DIR),
        "image_embeddings_shape": list(image_emb.shape),
        "image_meta_shape": list(image_meta.shape),
        "vlm_score_shape": list(scores.shape),
        "matched": int(n_match),
        "total": int(n_total),
        "match_rate": float(n_match / max(n_total, 1)),
        "duplicated_meta_keys": int(duplicated_meta_keys),
        "protocol_order": PROTOCOL_ORDER,
        "frame_embedding_dim": int(dim),
        "patient_embedding_concat_dim": int(X_patient.shape[1]),
        "n_patients": int(len(patient_ids)),
        "weight_formula": "final_weight_raw = attention_weight * vlm_reweight_factor; final_weight_norm = group_normalized(final_weight_raw) within patient_id + protocol",
        "note": "当前输入文件为 mock VLM score CSV；若 vlm_mode=mock，结果只能作为流程验证或 mock-VLM reweighting 结果。",
    }

    with open(OUT_DIR / "alignment_report.json", "w", encoding="utf-8") as f:
        json.dump(alignment_report, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print("输出目录:", OUT_DIR)
    print("主要输出:")
    print(" - patient_embeddings_concat.npy")
    print(" - patient_embeddings_pca.npy")
    print(" - patient_ids.txt")
    print(" - clusters.csv")
    print(" - cluster_metrics.json")
    print(" - alignment_report.json")
    print(" - topk_frame_scores_aligned.csv")
    print(" - protocol_pooling_stats.csv")


if __name__ == "__main__":
    main()
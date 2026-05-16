"""
batch_generate_embeddings.py

Final version.
Batch generation of patient-level embeddings based on
NPYFeatureLoader v2.0 (frame-level semantics).

- Mean Pooling: deterministic temporal truncation (MAPP)
- Attention Pooling: deterministic centroid-softmax top-k
- Strict protocol completeness enforced (paper-ready)

Author: you
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from backend.data_loader.npy_loader import NPYFeatureLoader



# =====================================================
# Utils
# =====================================================
def l2norm(x: np.ndarray, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def get_all_patient_ids(features_dir: Path):
    return sorted([d.name for d in features_dir.iterdir() if d.is_dir()])


# =====================================================
# Pooling functions (frame-level)
# =====================================================
def mean_pool_protocol(feats: np.ndarray) -> np.ndarray:
    """
    Deterministic mean pooling over frames.
    feats: (N, 768)
    """
    return feats.mean(axis=0)


def attention_pool_protocol(
    feats: np.ndarray,
    topk: int,
    temperature: float = 0.07
) -> np.ndarray:
    """
    Deterministic centroid-softmax attention pooling.
    No random sampling involved.

    feats: (N, 768)
    """
    if len(feats) == 0:
        return np.zeros((768,), dtype=np.float32)

    En = l2norm(feats, axis=1)
    centroid = l2norm(En.mean(axis=0, keepdims=True)).squeeze(0)
    scores = En @ centroid  # (N,)

    if topk > 0 and len(scores) > topk:
        idx = np.argsort(-scores)[:topk]
        feats = feats[idx]
        scores = scores[idx]

    w = softmax((scores / temperature).reshape(1, -1)).reshape(-1)
    z = (w[:, None] * feats).sum(axis=0)

    return z.astype(np.float32)


# =====================================================
# Main
# =====================================================
def main():
    # ---------- paths ----------
    features_dir = Path(r"D:\dataProcess\dinov2_features")
    out_dir = Path(r"D:\dataProcess\phenotyping_system\data\processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = get_all_patient_ids(features_dir)
    protocols = NPYFeatureLoader.PROTOCOL_ORDER

    print(f"📊 Found {len(patient_ids)} patients")

    # =================================================
    # Mean Pooling (frame-level MAPP = 6)
    # =================================================
    print("\n" + "=" * 70)
    print("📈 Mean Pooling (frame-level MAPP = 6, deterministic)")
    print("=" * 70)

    loader_mean = NPYFeatureLoader(
        features_dir=str(features_dir),
        mapp=6
    )

    X_mean = []
    kept_mean = []

    for pid in tqdm(patient_ids, desc="Mean Pooling"):
        try:
            protocols_feat = loader_mean.load_patient(
                pid,
                strict=True  # ✅ 强制协议完整性
            )
            vecs = [mean_pool_protocol(protocols_feat[p]) for p in protocols]
            X_mean.append(np.concatenate(vecs))
            kept_mean.append(pid)
        except Exception as e:
            tqdm.write(f"❌ Mean | {pid}: {e}")

    X_mean = np.stack(X_mean)

    np.savez(
        out_dir / "patient_embeddings_mean_mapp6.npz",
        patient_ids=np.array(kept_mean),
        embeddings=X_mean,
        protocols=protocols,
        config=dict(
            pooling="mean",
            mapp=6,
            deterministic=True,
            frame_level=True
        )
    )

    # =================================================
    # Attention Pooling (frame-level topk = 8)
    # =================================================
    print("\n" + "=" * 70)
    print("🧠 Attention Pooling (frame-level topk = 8, deterministic)")
    print("=" * 70)

    loader_attn = NPYFeatureLoader(
        features_dir=str(features_dir),
        mapp=None
    )

    X_attn = []
    kept_attn = []

    for pid in tqdm(patient_ids, desc="Attention Pooling"):
        try:
            protocols_feat = loader_attn.load_patient(
                pid,
                strict=True  # ✅ 强制协议完整性
            )
            vecs = [
                attention_pool_protocol(
                    protocols_feat[p],
                    topk=8,
                    temperature=0.07
                )
                for p in protocols
            ]
            X_attn.append(np.concatenate(vecs))
            kept_attn.append(pid)
        except Exception as e:
            tqdm.write(f"❌ Attn | {pid}: {e}")

    X_attn = np.stack(X_attn)

    np.savez(
        out_dir / "patient_embeddings_attention_topk8_tau007.npz",
        patient_ids=np.array(kept_attn),
        embeddings=X_attn,
        protocols=protocols,
        config=dict(
            pooling="attention",
            topk=8,
            temperature=0.07,
            deterministic=True,
            frame_level=True
        )
    )

    # =================================================
    # Summary
    # =================================================
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Mean Pooling:      {len(kept_mean)} / {len(patient_ids)} patients")
    print(f"Attention Pooling: {len(kept_attn)} / {len(patient_ids)} patients")
    print(f"Saved to: {out_dir}")
    print("✅ Batch embedding generation finished.")


if __name__ == "__main__":
    main()

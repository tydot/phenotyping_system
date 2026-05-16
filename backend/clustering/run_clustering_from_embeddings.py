"""
run_clustering_from_embeddings.py

System-level clustering pipeline for patient-level embeddings.

Supports:
- Mean Pooling embeddings
- Attention Pooling embeddings

Pipeline:
embedding -> standardize -> PCA -> KMeans -> metrics -> save

Author: you
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# =====================================================
# Utils
# =====================================================
def load_embeddings(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "patient_ids": data["patient_ids"],
        "embeddings": data["embeddings"],
        "protocols": list(data["protocols"]),
        "config": data["config"].item() if "config" in data else {},
    }


def save_clusters_csv(out_path: Path, patient_ids, labels):
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "cluster"])
        for pid, c in zip(patient_ids, labels):
            w.writerow([pid, int(c)])


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to .npz patient embeddings")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--pca_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_pca", action="store_true",
                        help="Disable PCA")
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load embeddings
    # -----------------------------
    data = load_embeddings(emb_path)
    X = data["embeddings"]
    patient_ids = data["patient_ids"]

    print(f"📊 Loaded embeddings: {X.shape}")
    print(f"Patients: {len(patient_ids)}")

    # -----------------------------
    # Standardize
    # -----------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # -----------------------------
    # PCA
    # -----------------------------
    if args.no_pca:
        Xp = Xs
        pca = None
        print("⚠️ PCA disabled")
    else:
        pca_dim = min(args.pca_dim, Xs.shape[1])
        pca = PCA(n_components=pca_dim, random_state=args.seed)
        Xp = pca.fit_transform(Xs)
        print(f"🔽 PCA: {Xs.shape[1]} -> {Xp.shape[1]}")

    # -----------------------------
    # Clustering
    # -----------------------------
    km = KMeans(
        n_clusters=args.n_clusters,
        n_init="auto",
        random_state=args.seed
    )
    labels = km.fit_predict(Xp)

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "n_patients": int(len(patient_ids)),
        "n_clusters": int(args.n_clusters),
        "seed": int(args.seed),
        "pca_enabled": not args.no_pca,
        "pca_dim": int(Xp.shape[1]),
        "silhouette": float(silhouette_score(Xp, labels)),
        "davies_bouldin": float(davies_bouldin_score(Xp, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Xp, labels)),
        "embedding_config": data["config"],
    }

    # -----------------------------
    # Save outputs
    # -----------------------------
    save_clusters_csv(out_dir / "clusters.csv", patient_ids, labels)

    with open(out_dir / "cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if pca is not None:
        np.save(out_dir / "pca_components.npy", pca.components_)

    print("✅ Clustering finished")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

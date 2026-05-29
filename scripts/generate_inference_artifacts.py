"""
generate_inference_artifacts.py

生成在线推理所需的离线产物到 outputs/inference_artifacts/。

产物清单：
  - scaler_mean.npy / scaler_scale.npy   — StandardScaler 参数
  - pca_components.npy / pca_mean.npy     — PCA 降维矩阵
  - cluster_prototypes_pca.npy            — 各 cluster 在 PCA 空间的质心
  - reference_patient_embeddings_pca.csv  — 所有患者的 PCA 嵌入 + cluster 标签

数据来源：
  - 嵌入：outputs/m6_attn_vlm_topk4_mock_from_cache_k3/patient_embeddings_concat.npy
  - Scaler/PCA：outputs/m6_attn_vlm_topk4_mock_from_cache_k3/{scaler,pca}_*.npy
  - Cluster 标签：processed/M1/consensus_labels.csv

用法：python scripts/generate_inference_artifacts.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# === 路径配置 ===
M6_DIR = ROOT_DIR / "outputs" / "m6_attn_vlm_topk4_mock_from_cache_k3"
CONSENSUS_PATH = ROOT_DIR / "processed" / "M1" / "consensus_labels.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "inference_artifacts"


def main():
    print("=" * 60)
    print("  生成在线推理产物")
    print("=" * 60)

    # --- 1. 加载已有 Scaler/PCA 参数 ---
    print("\n[1/5] 加载 Scaler 和 PCA 参数...")

    scaler_mean = np.load(M6_DIR / "scaler_mean.npy")
    scaler_scale = np.load(M6_DIR / "scaler_scale.npy")
    pca_components = np.load(M6_DIR / "pca_components.npy")
    pca_mean = np.load(M6_DIR / "pca_mean.npy")

    print(f"  Scaler mean:  {scaler_mean.shape}")
    print(f"  Scaler scale: {scaler_scale.shape}")
    print(f"  PCA components: {pca_components.shape}")
    print(f"  PCA mean: {pca_mean.shape}")

    # --- 2. 加载嵌入和标签 ---
    print("\n[2/5] 加载嵌入和 consensus 标签...")

    embeddings = np.load(M6_DIR / "patient_embeddings_concat.npy")
    m6_clusters = pd.read_csv(M6_DIR / "clusters.csv")
    m1_labels = pd.read_csv(CONSENSUS_PATH)

    print(f"  嵌入矩阵: {embeddings.shape}")
    print(f"  M6 患者数: {len(m6_clusters)}")
    print(f"  M1 consensus 患者数: {len(m1_labels)}")

    # --- 3. 对齐患者顺序 ---
    print("\n[3/5] 对齐患者顺序...")

    # M6 clusters.csv 的 patient_id 顺序对应 embeddings 行顺序
    m6_clusters["patient_id"] = m6_clusters["patient_id"].astype(str)
    m1_labels["patient_id"] = m1_labels["patient_id"].astype(str)

    # 按 M6 顺序建立索引
    m6_id_to_idx = {pid: i for i, pid in enumerate(m6_clusters["patient_id"])}

    # 对齐 M1 labels 到 M6 顺序
    aligned_labels = []
    aligned_indices = []
    for _, row in m6_clusters.iterrows():
        pid = row["patient_id"]
        m1_row = m1_labels[m1_labels["patient_id"] == pid]
        if len(m1_row) == 0:
            print(f"  [WARN] 患者 {pid} 在 M1 consensus 中未找到，跳过")
            continue
        aligned_labels.append(m1_row.iloc[0]["consensus_cluster"])
        aligned_indices.append(m6_id_to_idx[pid])

    aligned_embeddings = embeddings[aligned_indices]
    aligned_labels = np.array(aligned_labels)
    aligned_ids = m6_clusters.iloc[aligned_indices]["patient_id"].values

    print(f"  对齐后样本数: {len(aligned_labels)}")
    print(f"  Cluster 分布: {dict(zip(*np.unique(aligned_labels, return_counts=True)))}")

    # --- 4. PCA 变换 ---
    print("\n[4/5] 执行 PCA 变换...")

    # 标准化
    X_scaled = (aligned_embeddings - scaler_mean) / (scaler_scale + 1e-12)

    # PCA 投影
    X_pca = (X_scaled - pca_mean) @ pca_components.T

    print(f"  PCA 后维度: {X_pca.shape}")

    # 计算 cluster 质心
    unique_clusters = sorted(np.unique(aligned_labels))
    prototypes = []
    for c in unique_clusters:
        mask = aligned_labels == c
        centroid = X_pca[mask].mean(axis=0)
        prototypes.append(centroid)
        print(f"  Cluster {c}: {mask.sum()} 患者, 质心范数={np.linalg.norm(centroid):.4f}")

    prototypes = np.stack(prototypes, axis=0)

    # --- 5. 保存产物 ---
    print("\n[5/5] 保存产物...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "scaler_mean.npy", scaler_mean)
    np.save(OUTPUT_DIR / "scaler_scale.npy", scaler_scale)
    np.save(OUTPUT_DIR / "pca_components.npy", pca_components)
    np.save(OUTPUT_DIR / "pca_mean.npy", pca_mean)
    np.save(OUTPUT_DIR / "cluster_prototypes_pca.npy", prototypes)

    # 生成 reference CSV
    pc_cols = [f"pc{i+1}" for i in range(X_pca.shape[1])]
    ref_df = pd.DataFrame(X_pca, columns=pc_cols)
    ref_df.insert(0, "patient_id", aligned_ids)
    ref_df.insert(1, "consensus_cluster", aligned_labels)
    ref_df.to_csv(OUTPUT_DIR / "reference_patient_embeddings_pca.csv", index=False)

    # --- 验证 ---
    print("\n" + "=" * 60)
    print("  产物验证")
    print("=" * 60)

    expected_files = [
        "scaler_mean.npy",
        "scaler_scale.npy",
        "pca_components.npy",
        "pca_mean.npy",
        "cluster_prototypes_pca.npy",
        "reference_patient_embeddings_pca.csv",
    ]

    all_ok = True
    for fname in expected_files:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  [OK] {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  [FAIL] {fname} 未生成!")
            all_ok = False

    if all_ok:
        print(f"\n全部产物已生成到: {OUTPUT_DIR}")
    else:
        print(f"\n部分产物缺失，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()

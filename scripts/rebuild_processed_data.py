"""
rebuild_processed_data.py

从原始聚类输出重新生成 processed/M1-M5 数据。

数据源：
  - 公平方法：dataProcess/outputs/fair_compare_multi_dim
  - 旧随机方法：dataProcess/rerun_outputs_02468_pca50
  - 临床表：dataProcess/report_valid_only_clean.csv

用法：python scripts/rebuild_processed_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT_DIR = Path(__file__).resolve().parents[1]

# === 数据路径 ===
DATA_PROCESS_DIR = Path(r"H:\windows\image_data\dataProcess")
FAIR_DIR = DATA_PROCESS_DIR / "outputs" / "fair_compare_multi_dim"
OLD_DIR = DATA_PROCESS_DIR / "rerun_outputs_02468_pca50"
CLINICAL_PATH = DATA_PROCESS_DIR / "report_valid_only_clean.csv"
OUTPUT_DIR = ROOT_DIR / "processed"

# === 版本配置 ===
VERSIONS = {
    "M1": {
        "display_name": "M1 - fair Attention k=4",
        "method": "fair",
        "pooling": "attention",
        "top_k": 4,
        "source": "consensus_labels.csv",
        "path": FAIR_DIR / "attn" / "topk4" / "pca50",
    },
    "M2": {
        "display_name": "M2 - fair Attention k=6",
        "method": "fair",
        "pooling": "attention",
        "top_k": 6,
        "source": "consensus_labels_all.csv",
        "path": FAIR_DIR / "attn" / "topk6" / "pca50",
    },
    "M3": {
        "display_name": "M3 - fair Mean all",
        "method": "fair",
        "pooling": "mean",
        "top_k": 0,
        "source": "consensus_labels.csv",
        "path": FAIR_DIR / "mean" / "topk0" / "pca50",
    },
    "M4": {
        "display_name": "M4 - old random Attention k=4",
        "method": "old_random",
        "pooling": "attention",
        "top_k": 4,
        "source": "seed_consensus",
        "path": OLD_DIR / "attn" / "topk4",
    },
    "M5": {
        "display_name": "M5 - old random Mean mapp6",
        "method": "old_random",
        "pooling": "mean",
        "top_k": 6,
        "source": "clusters_consensus.csv",
        "path": OLD_DIR / "mean" / "mapp6",
    },
}

N_CLUSTERS = 3
CONFIDENCE_THRESHOLD = 0.8


# =====================================================
# 工具函数
# =====================================================

def normalize_pid(pid) -> str:
    """标准化患者 ID：去掉 .0 后缀"""
    s = str(pid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def align_labels(ref_labels: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
    """
    用匈牙利算法对齐 target_labels 到 ref_labels 的标签空间。
    """
    n_classes = max(ref_labels.max(), target_labels.max()) + 1
    cost_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(ref_labels)):
        r = int(ref_labels[i])
        t = int(target_labels[i])
        cost_matrix[r, t] += 1

    # 最大化匹配 = 最小化负值
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    mapping = np.zeros(n_classes, dtype=int)
    for r, c in zip(row_ind, col_ind):
        mapping[c] = r

    return mapping[target_labels]


def generate_consensus_from_seeds(seed_dir: Path, n_seeds: int = 10) -> pd.DataFrame:
    """
    从多个 seed 的 clusters.csv 生成共识标签。
    """
    all_labels = []
    patient_ids = None

    for i in range(n_seeds):
        seed_path = seed_dir / f"seed{i}" / "clusters.csv"
        if not seed_path.exists():
            print(f"  [WARN] {seed_path} 不存在，跳过")
            continue

        df = pd.read_csv(seed_path)
        df["patient_id"] = df["patient_id"].astype(str)

        if patient_ids is None:
            patient_ids = df["patient_id"].values

        all_labels.append(df["cluster"].values)

    if not all_labels:
        raise ValueError(f"没有找到任何 seed 数据: {seed_dir}")

    n_patients = len(patient_ids)
    n_seeds_loaded = len(all_labels)

    # 以 seed0 为基准，对齐所有 seed
    ref = all_labels[0]
    aligned = np.zeros((n_patients, n_seeds_loaded), dtype=int)
    aligned[:, 0] = ref

    for i in range(1, n_seeds_loaded):
        aligned[:, i] = align_labels(ref, all_labels[i])

    # 投票
    consensus = np.zeros(n_patients, dtype=int)
    confidence = np.zeros(n_patients, dtype=float)

    for j in range(n_patients):
        votes = aligned[j, :]
        counts = np.bincount(votes, minlength=N_CLUSTERS)
        consensus[j] = np.argmax(counts)
        confidence[j] = counts[consensus[j]] / n_seeds_loaded

    # 计算 switch_rate（各 seed 与共识不一致的比例）
    switch_rate = np.zeros(n_patients, dtype=float)
    for j in range(n_patients):
        switch_rate[j] = np.mean(aligned[j, :] != consensus[j])

    return pd.DataFrame({
        "patient_id": patient_ids,
        "pid_key": patient_ids,
        "consensus_cluster": consensus,
        "confidence": confidence,
        "switch_rate": switch_rate,
    })


def load_fair_consensus(path: Path, source_file: str) -> pd.DataFrame:
    """加载公平方法的共识标签文件。"""
    df = pd.read_csv(path / source_file)
    df["patient_id"] = df["patient_id"].astype(str)
    df["pid_key"] = df["pid_key"].astype(str)

    # 只保留需要的列
    cols = ["patient_id", "pid_key", "consensus_cluster", "confidence", "switch_rate"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0

    return df[cols]


def load_m5_consensus(path: Path) -> pd.DataFrame:
    """M5 共识：忽略 clusters_consensus.csv（标签空间不一致），直接从 10 个 seed 投票。"""
    return generate_consensus_from_seeds(path)


# =====================================================
# 主流程
# =====================================================

def main():
    print("=" * 60)
    print("  重新生成 M1-M5 分型数据")
    print("=" * 60)

    # --- 1. 加载临床表 ---
    print("\n[1/3] 加载临床表...")
    clinical = pd.read_csv(CLINICAL_PATH, encoding="utf-8-sig")
    clinical["_pid_norm"] = clinical["_pid_norm"].apply(normalize_pid)
    clinical_pids = set(clinical["_pid_norm"])
    print(f"  临床表: {len(clinical)} 患者")

    # --- 2. 加载各版本共识标签 ---
    print("\n[2/3] 加载各版本共识标签...")

    all_consensus = {}

    for ver_name, ver_conf in VERSIONS.items():
        print(f"\n  --- {ver_name}: {ver_conf['display_name']} ---")

        if ver_conf["source"] == "consensus_labels.csv":
            df = load_fair_consensus(ver_conf["path"], "consensus_labels.csv")
        elif ver_conf["source"] == "consensus_labels_all.csv":
            df = load_fair_consensus(ver_conf["path"], "consensus_labels_all.csv")
        elif ver_conf["source"] == "seed_consensus":
            df = generate_consensus_from_seeds(ver_conf["path"])
        elif ver_conf["source"] == "clusters_consensus.csv":
            df = load_m5_consensus(ver_conf["path"])
        else:
            print(f"  [ERROR] 未知 source: {ver_conf['source']}")
            continue

        df["patient_id"] = df["patient_id"].astype(str).apply(normalize_pid)
        df["pid_key"] = df["pid_key"].astype(str).apply(normalize_pid)

        # 对齐到临床表
        df_aligned = df[df["patient_id"].isin(clinical_pids)].copy()
        df_aligned = df_aligned.drop_duplicates(subset=["patient_id"], keep="first")

        all_consensus[ver_name] = df_aligned

        # 统计
        n = len(df_aligned)
        cluster_dist = df_aligned["consensus_cluster"].value_counts().sort_index()
        boundary = (df_aligned["confidence"] < CONFIDENCE_THRESHOLD).sum()

        print(f"  对齐后: {n} 患者")
        print(f"  Cluster 分布: {cluster_dist.to_dict()}")
        print(f"  边界 (<{CONFIDENCE_THRESHOLD}): {boundary} ({boundary/n*100:.1f}%)")
        print(f"  Confidence: min={df_aligned['confidence'].min():.4f}, mean={df_aligned['confidence'].mean():.4f}")

    # --- 3. 生成输出文件 ---
    print("\n[3/3] 生成输出文件...")

    for ver_name, ver_conf in VERSIONS.items():
        if ver_name not in all_consensus:
            continue

        df_consensus = all_consensus[ver_name]
        out_dir = OUTPUT_DIR / ver_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 保存 consensus_labels.csv
        consensus_out = df_consensus[["patient_id", "pid_key", "consensus_cluster", "confidence", "switch_rate"]].copy()
        consensus_out.to_csv(out_dir / "consensus_labels.csv", index=False, encoding="utf-8-sig")

        # 合并临床数据
        clinical_copy = clinical.copy()
        clinical_copy = clinical_copy.rename(columns={"_pid_norm": "patient_id"})

        merged = df_consensus.merge(
            clinical_copy,
            on="patient_id",
            how="left",
            suffixes=("", "_clinical"),
        )

        # 添加 stability_label
        merged["is_boundary"] = (merged["confidence"] < CONFIDENCE_THRESHOLD).astype(int)
        merged["stability_label"] = merged["is_boundary"].map({0: "stable", 1: "boundary"})

        # 保存 merged_clinical_all.csv
        merged.to_csv(out_dir / "merged_clinical_all.csv", index=False, encoding="utf-8-sig")

        print(f"  {ver_name}: consensus_labels.csv ({len(consensus_out)} rows), merged_clinical_all.csv ({len(merged)} rows)")

    # --- 4. 汇总统计 ---
    print("\n" + "=" * 60)
    print("  汇总统计")
    print("=" * 60)

    summary_rows = []
    for ver_name, ver_conf in VERSIONS.items():
        if ver_name not in all_consensus:
            continue

        df = all_consensus[ver_name]
        n = len(df)
        boundary = (df["confidence"] < CONFIDENCE_THRESHOLD).sum()

        summary_rows.append({
            "version": ver_name,
            "display_name": ver_conf["display_name"],
            "method": ver_conf["method"],
            "pooling": ver_conf["pooling"],
            "top_k": ver_conf["top_k"],
            "n_patients": n,
            "n_boundary": boundary,
            "boundary_ratio": f"{boundary/n*100:.1f}%",
            "confidence_min": f"{df['confidence'].min():.4f}",
            "confidence_mean": f"{df['confidence'].mean():.4f}",
            "cluster_0": int((df["consensus_cluster"] == 0).sum()),
            "cluster_1": int((df["consensus_cluster"] == 1).sum()),
            "cluster_2": int((df["consensus_cluster"] == 2).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存到: {OUTPUT_DIR / 'summary.csv'}")
    print("\n完成!")


if __name__ == "__main__":
    main()

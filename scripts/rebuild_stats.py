"""
rebuild_stats.py

为 M1-M5 重新生成 Kruskal-Wallis 和 Dunn 检验统计文件。

用法：python scripts/rebuild_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from itertools import combinations

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "processed"

VERSIONS = {
    "M1": "M1_新公平_Attn_topk4",
    "M2": "M2_新公平_Attn_topk6",
    "M3": "M3_新公平_Mean_all",
    "M4": "M4_旧随机_Attn_topk4",
    "M5": "M5_旧随机_Mean_mapp6",
}

CONFIDENCE_THRESHOLD = 0.8

METRICS = [
    '肛门括约肌静息压(mmHg)',
    '最大缩榨压MSP（mmHg）',
    '排便时直肠压力(mmHg)',
    '初始感觉阈值(ml)',
    '初始便意阈值(ml)',
    '排便窘迫感阈值(ml)',
    '最大容量感觉阈值(ml)',
    'RAIR诱发最小容积(ml)',
]


def run_kruskal(df, metrics, cluster_col='consensus_cluster'):
    results = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        groups = []
        valid_clusters = []
        for c in sorted(df[cluster_col].dropna().unique()):
            vals = df[df[cluster_col] == c][metric].dropna()
            if len(vals) >= 5:
                groups.append(vals)
                valid_clusters.append(c)
        if len(groups) < 2:
            continue

        H, p_raw = kruskal(*groups)
        n = sum(len(g) for g in groups)
        k = len(groups)
        epsilon_sq = H / (n - 1) if n > 1 else 0

        # Holm 校正（简化版）
        p_adj = min(p_raw * len(metrics), 1.0)

        sig_raw = '***' if p_raw < 0.001 else '**' if p_raw < 0.01 else '*' if p_raw < 0.05 else 'ns'
        sig_adj = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'

        results.append({
            '指标': metric,
            '样本量': n,
            'H': round(H, 4),
            'p_raw': p_raw,
            'epsilon_squared': round(epsilon_sq, 6),
            'p_adj_holm': p_adj,
            '显著性_raw': sig_raw,
            '显著性_adj': sig_adj,
            '是否显著': p_raw < 0.05,
        })

    return pd.DataFrame(results)


def run_dunn(df, metrics, cluster_col='consensus_cluster'):
    results = []
    clusters = sorted(df[cluster_col].dropna().unique())

    for metric in metrics:
        if metric not in df.columns:
            continue

        # 收集各 cluster 数据
        cluster_data = {}
        for c in clusters:
            vals = df[df[cluster_col] == c][metric].dropna()
            if len(vals) >= 5:
                cluster_data[c] = vals

        if len(cluster_data) < 2:
            continue

        # 两两比较（简化版 Dunn：Mann-Whitney U + Holm 校正）
        from scipy.stats import mannwhitneyu
        pairs = list(combinations(sorted(cluster_data.keys()), 2))
        n_tests = len(pairs)

        for i, (c1, c2) in enumerate(pairs):
            U, p = mannwhitneyu(cluster_data[c1], cluster_data[c2], alternative='two-sided')
            p_adj = min(p * n_tests, 1.0)
            sig = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'

            results.append({
                '指标': metric,
                '对比': f'{c1} vs {c2}',
                'p_adj': round(p_adj, 6),
                '显著性': sig,
            })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("  重新生成 M1-M5 Kruskal/Dunn 统计")
    print("=" * 60)

    for ver, ver_name in VERSIONS.items():
        print(f"\n  === {ver} ({ver_name}) ===")

        # 加载数据
        df = pd.read_csv(OUTPUT_DIR / ver / "merged_clinical_all.csv")

        # 确保 consensus_cluster 是数值
        df['consensus_cluster'] = pd.to_numeric(df['consensus_cluster'], errors='coerce')

        stats_dir = OUTPUT_DIR / ver / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # 全部患者
        kruskal_all = run_kruskal(df, METRICS)
        kruskal_all['版本'] = ver_name
        kruskal_all.to_csv(stats_dir / f"{ver_name}_kruskal.csv", index=False, encoding='utf-8-sig')
        print(f"  kruskal_all: {len(kruskal_all)} metrics, n={len(df)}")

        dunn_all = run_dunn(df, METRICS)
        dunn_all['版本'] = ver_name
        dunn_all.to_csv(stats_dir / f"{ver_name}_dunn.csv", index=False, encoding='utf-8-sig')
        print(f"  dunn_all: {len(dunn_all)} comparisons")

    print("\n完成!")


if __name__ == "__main__":
    main()

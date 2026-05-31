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
    '肛门括约肌长度(cm)',
    '缩肛持续时间(s)',
    '排便时直肠压力(mmHg)',
    '最大容量感觉阈值(ml)',
    '初始感觉阈值(ml)',
    '初始便意阈值(ml)',
    '排便窘迫感阈值(ml)',
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

        # 暂存，后续统一 Holm 校正
        results.append({
            '指标': metric,
            '样本量': n,
            'H': round(H, 4),
            'p_raw': p_raw,
            'epsilon_squared': round(epsilon_sq, 6),
        })

    # 正确的 Holm step-down 校正（含单调性保证）
    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        pvals = res_df['p_raw'].values
        n_tests = len(pvals)
        sorted_idx = np.argsort(pvals)
        p_adj_holm = np.ones(n_tests)
        for rank, idx in enumerate(sorted_idx):
            p_adj_holm[idx] = min(pvals[idx] * (n_tests - rank), 1.0)
        # 单调性修正：确保排序后的 adjusted p 单调递增
        p_adj_holm[sorted_idx] = np.maximum.accumulate(p_adj_holm[sorted_idx])
        res_df['p_adj_holm'] = p_adj_holm
        res_df['显著性_raw'] = res_df['p_raw'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
        res_df['显著性_adj'] = res_df['p_adj_holm'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
        res_df['是否显著'] = res_df['p_adj_holm'] < 0.05

    return res_df


def run_dunn(df, metrics, cluster_col='consensus_cluster'):
    """真正的 Dunn 事后检验（使用 scikit-posthocs）。"""
    try:
        import scikit_posthocs as posthoc
    except ImportError:
        raise RuntimeError("请安装 scikit-posthocs: pip install scikit-posthocs")

    results = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        sub = df[[cluster_col, metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna()

        clusters = sorted(sub[cluster_col].unique())
        if len(clusters) < 2:
            continue

        mat = posthoc.posthoc_dunn(sub, val_col=metric, group_col=cluster_col, p_adjust="holm")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                p_adj = float(mat.iloc[i, j])
                c1, c2 = int(clusters[i]), int(clusters[j])
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

        # 稳定患者（confidence >= 0.8）
        df_stable = df[df['confidence'] >= CONFIDENCE_THRESHOLD].copy()

        stats_dir = OUTPUT_DIR / ver / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # 稳定患者 Kruskal
        kruskal_stable = run_kruskal(df_stable, METRICS)
        kruskal_stable['版本'] = ver_name
        kruskal_stable.to_csv(stats_dir / f"{ver_name}_kruskal.csv", index=False, encoding='utf-8-sig')
        n_sig = kruskal_stable['是否显著'].sum() if len(kruskal_stable) > 0 else 0
        print(f"  kruskal_stable: {len(kruskal_stable)} metrics, n={len(df_stable)}, sig={n_sig}")

        dunn_stable = run_dunn(df_stable, METRICS)
        dunn_stable['版本'] = ver_name
        dunn_stable.to_csv(stats_dir / f"{ver_name}_dunn.csv", index=False, encoding='utf-8-sig')
        print(f"  dunn_stable: {len(dunn_stable)} comparisons")

    print("\n完成!")


if __name__ == "__main__":
    main()

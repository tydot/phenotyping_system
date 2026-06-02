"""
rebuild_rair_analysis.py

为 M1-M5 重新生成 RAIR 分析文件。
- 从 RAIR 特征表 + 新共识标签合并
- 运行 Kruskal-Wallis 检验

用法：python scripts/rebuild_rair_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "processed"
RAIR_FEATURES_PATH = Path(r"H:\windows\image_data\dataProcess\downstream\rair_validation\rair_features.csv")

VERSIONS = ["M1", "M2", "M3", "M4", "M5"]
CONFIDENCE_THRESHOLD = 0.8


def normalize_pid(x):
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def run_kruskal_rair(df, tag, out_dir):
    """对 RAIR relaxation_amplitude 进行 Kruskal-Wallis 检验。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    if df['consensus_cluster'].nunique() < 2:
        print(f"  {tag}: <2 clusters, skip")
        return None

    groups = []
    for _, g in df.groupby('consensus_cluster'):
        vals = g['relaxation_amplitude'].dropna().values
        if len(vals) >= 5:
            groups.append(vals)

    if len(groups) < 2:
        print(f"  {tag}: <2 valid clusters (min 5 per cluster), skip")
        return None

    H, p = kruskal(*groups)

    stat_row = {
        'subset': tag,
        'n_patients': len(df),
        'n_clusters': df['consensus_cluster'].nunique(),
        'H_statistic': round(H, 4),
        'p_value': p,
        'significant': p < 0.05,
    }

    # 绘制 boxplot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(6, 4), dpi=150)
    ax = sns.boxplot(data=df, x='consensus_cluster', y='relaxation_amplitude', showfliers=False,
                     palette=['#E63946', '#457B9D', '#2A9D8F'])
    sns.stripplot(data=df, x='consensus_cluster', y='relaxation_amplitude',
                  color='black', alpha=0.4, size=2)
    tag_cn = '全部患者' if tag == 'all' else '稳定患者'
    plt.title(f'RAIR 松弛幅度分布（{tag_cn}）')
    plt.xlabel('共识簇')
    plt.ylabel('松弛幅度（代理信号）')
    plt.tight_layout()
    plt.savefig(out_dir / f'rair_amplitude_by_cluster_{tag}.png', dpi=300)
    plt.close()

    return stat_row


def main():
    print("=" * 60)
    print("  重新生成 M1-M5 RAIR 分析")
    print("=" * 60)

    # 加载 RAIR 特征表
    rair = pd.read_csv(RAIR_FEATURES_PATH)
    rair['patient_id'] = rair['patient_id'].astype(str).apply(normalize_pid)
    print(f"  RAIR 特征表: {len(rair)} 患者")

    for ver in VERSIONS:
        print(f"\n  === {ver} ===")

        # 加载新共识标签
        consensus = pd.read_csv(OUTPUT_DIR / ver / "consensus_labels.csv")
        consensus['patient_id'] = consensus['patient_id'].astype(str).apply(normalize_pid)

        # 合并
        merged = rair.merge(
            consensus[['patient_id', 'consensus_cluster', 'confidence']],
            on='patient_id',
            how='inner'
        )
        print(f"  合并后: {len(merged)} 患者")

        # 保存
        ver_dir = OUTPUT_DIR / ver / "rair_analysis"
        ver_dir.mkdir(parents=True, exist_ok=True)

        merged.to_csv(ver_dir / f"rair_with_clusters_{ver}.csv", index=False, encoding='utf-8-sig')

        # Kruskal-Wallis 检验
        stats = []

        # 全部患者
        s = run_kruskal_rair(merged, "all", ver_dir / "all")
        if s:
            s['version'] = ver
            stats.append(s)

        # 稳定患者
        stable = merged[merged['confidence'] >= CONFIDENCE_THRESHOLD].copy()
        if len(stable) > 0:
            s = run_kruskal_rair(stable, "stable", ver_dir / "stable")
            if s:
                s['version'] = ver
                stats.append(s)

        # 保存统计结果
        if stats:
            pd.DataFrame(stats).to_csv(ver_dir / "rair_cluster_stats.csv", index=False, encoding='utf-8-sig')

        # 打印结果
        for s in stats:
            sig = '***' if s['p_value'] < 0.001 else '**' if s['p_value'] < 0.01 else '*' if s['p_value'] < 0.05 else 'ns'
            print(f"  {s['subset']}: n={s['n_patients']}, H={s['H_statistic']:.4f}, p={s['p_value']:.4e} {sig}")

    print("\n完成!")


if __name__ == "__main__":
    main()

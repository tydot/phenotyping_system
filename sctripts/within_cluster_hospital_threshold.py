# -*- coding: utf-8 -*-
"""
M1-M5 组内医院参考值偏离分析
单样本 Wilcoxon + Holm 校正

输入：
H:/windows/图像数据/dataProcess/processed/M*/merged_clinical_all.csv

输出：
H:/windows/图像数据/dataProcess/processed/within_cluster_results/
    within_cluster_all_by_hospital_threshold.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================

BASE_DIR = Path(r"H:\windows\图像数据\dataProcess\processed")

OUT_DIR = BASE_DIR / "within_cluster_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERSION_CONFIGS = {
    "M1_Attn_topk4": BASE_DIR / "M1" / "merged_clinical_all.csv",
    "M2_Attn_topk6": BASE_DIR / "M2" / "merged_clinical_all.csv",
    "M3_Mean_topk0": BASE_DIR / "M3" / "merged_clinical_all.csv",
    "M4_旧Attn_topk4": BASE_DIR / "M4" / "merged_clinical_all.csv",
    "M5_旧Mean_topk6": BASE_DIR / "M5" / "merged_clinical_all.csv",
}


# ============================================================
# 1. 参数
# ============================================================

CONFIDENCE_THRESHOLD = 0.8
MIN_N = 10
ALPHA = 0.05
CLUSTER_COL = "consensus_cluster"


# ============================================================
# 2. 医院报告参考值
# low, high, center
# ============================================================

REFERENCE = {
    "肛门括约肌静息压(mmHg)": {
        "M": (59, 115, 87),
        "F": (47, 101, 74),
    },
    "最大缩榨压MSP（mmHg）": {
        "M": (91, 170, 130.5),
        "F": (61, 140, 100.5),
    },
    "最大容量感觉阈值(ml)": {
        "ALL": (155, 309, 232),
    },
    "肛门括约肌长度(cm)": {
        "M": (3.4, 5.9, 4.65),
        "F": (2.7, 5.1, 3.9),
    },
    "缩肛持续时间(s)": {
        "ALL": (12.2, 14.4, 13.3),
    },
    "排便时直肠压力(mmHg)": {
        "ALL": (45, np.inf, 45),
    },
    "RAIR诱发最小容积(ml)": {
        "ALL": (0, 30, 30),
    },
    "初始感觉阈值(ml)": {
        "ALL": (0, 30, 30),
    },
    "初始便意阈值(ml)": {
        "ALL": (57, 196, 126.5),
    },
    "排便窘迫感阈值(ml)": {
        "ALL": (93, 241, 167),
    },
}


# ============================================================
# 3. 工具函数
# ============================================================

def normalize_sex(x):
    if pd.isna(x):
        return None

    s = str(x).strip().upper()

    return {
        "男": "M",
        "女": "F",
        "M": "M",
        "F": "F",
        "MALE": "M",
        "FEMALE": "F",
        "1": "M",
        "2": "F",
    }.get(s, None)


def p_to_star(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def wilcoxon_safe(vals, ref):
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values

    if len(v) < MIN_N:
        return np.nan

    diff = v - ref

    if np.allclose(diff, 0):
        return 1.0

    try:
        return wilcoxon(
            diff,
            zero_method="wilcox",
            alternative="two-sided"
        )[1]
    except Exception:
        return np.nan


def abnormal_ratio(vals, low, high):
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values

    if len(v) == 0:
        return np.nan

    if np.isinf(high):
        return float(np.mean(v < low))

    return float(np.mean((v < low) | (v > high)))


def direction_by_median(vals, ref_val):
    med = np.median(vals)
    if med > ref_val:
        return "高"
    if med < ref_val:
        return "低"
    return "等于参考中心"


# ============================================================
# 4. 主分析
# ============================================================

all_rows = []
version_overview_rows = []

for version_name, csv_path in VERSION_CONFIGS.items():
    print(f"\n▶ {version_name}")
    print(f"路径：{csv_path}")

    if not csv_path.exists():
        print("  ❌ 文件不存在")
        continue

    df = pd.read_csv(csv_path)

    total_n = len(df)

    if "confidence" in df.columns:
        df = df[df["confidence"] >= CONFIDENCE_THRESHOLD].copy()

    stable_n = len(df)

    if df.empty:
        print("  ❌ confidence筛选后无样本")
        continue

    if CLUSTER_COL not in df.columns:
        print(f"  ❌ 缺少 {CLUSTER_COL}")
        continue

    df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)

    sex_col = next((c for c in ["性别", "gender", "sex"] if c in df.columns), None)

    if sex_col:
        df["_sex_std"] = df[sex_col].apply(normalize_sex)
    else:
        df["_sex_std"] = None

    version_overview_rows.append({
        "版本": version_name,
        "总样本数": total_n,
        f"confidence>={CONFIDENCE_THRESHOLD}样本数": stable_n,
        "保留比例": round(stable_n / total_n, 4) if total_n else np.nan,
        "男性样本数": int((df["_sex_std"] == "M").sum()),
        "女性样本数": int((df["_sex_std"] == "F").sum()),
        "性别缺失或无法识别": int(df["_sex_std"].isna().sum()),
        "Cluster数": df[CLUSTER_COL].nunique(),
    })

    print(f"  总样本数：{total_n}")
    print(f"  稳定样本数：{stable_n}")
    print(f"  保留比例：{stable_n / total_n:.2%}")
    print("  Cluster分布：")
    print(df[CLUSTER_COL].value_counts().sort_index().to_string())

    for cluster in sorted(df[CLUSTER_COL].unique()):
        df_c = df[df[CLUSTER_COL] == cluster].copy()

        cluster_total_n = len(df_c)
        cluster_male_n = int((df_c["_sex_std"] == "M").sum())
        cluster_female_n = int((df_c["_sex_std"] == "F").sum())

        for metric, ref_def in REFERENCE.items():
            if metric not in df_c.columns:
                continue

            for sex_key, ref_tuple in ref_def.items():
                low, high, ref_val = ref_tuple

                if sex_key == "ALL":
                    sub = df_c.copy()
                else:
                    sub = df_c[df_c["_sex_std"] == sex_key].copy()

                if sub.empty:
                    continue

                sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
                sub = sub.dropna(subset=[metric])

                if len(sub) < MIN_N:
                    continue

                vals = sub[metric].values
                med = round(float(np.median(vals)), 2)
                mean_val = round(float(np.mean(vals)), 2)

                p_raw = wilcoxon_safe(vals, ref_val)

                all_rows.append({
                    "版本": version_name,
                    "Cluster": int(cluster),
                    "Cluster总样本数": cluster_total_n,
                    "Cluster男性样本数": cluster_male_n,
                    "Cluster女性样本数": cluster_female_n,
                    "指标": metric,
                    "性别亚组": sex_key,
                    "n": len(vals),
                    "均值": mean_val,
                    "中位数": med,
                    "参考下限": low,
                    "参考上限": high,
                    "参考中心": ref_val,
                    "偏离方向": direction_by_median(vals, ref_val),
                    "异常比例": round(abnormal_ratio(vals, low, high), 3),
                    "p_raw": p_raw,
                })


# ============================================================
# 5. 多重校正与输出
# ============================================================

res = pd.DataFrame(all_rows)

if res.empty:
    print("\n❌ 未生成任何统计结果，请检查指标名、样本量或路径。")
else:
    pvals = res["p_raw"].fillna(1.0).values
    res["p_adj"] = multipletests(
        pvals,
        alpha=ALPHA,
        method="holm"
    )[1]

    res["显著性"] = res["p_adj"].apply(p_to_star)
    res["显著"] = res["p_adj"] < ALPHA

    output_file = OUT_DIR / "within_cluster_all_by_hospital_threshold.csv"
    res.to_csv(output_file, index=False, encoding="utf-8-sig")

    overview = pd.DataFrame(version_overview_rows)
    overview_file = OUT_DIR / "version_sample_overview.csv"
    overview.to_csv(overview_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("医院阈值版组内偏离分析完成")
    print("=" * 80)
    print(f"明细结果：{output_file}")
    print(f"样本概况：{overview_file}")

    print("\n各版本显著偏离项数量：")
    sig_summary = (
        res.groupby("版本")["显著"]
        .sum()
        .reset_index()
        .rename(columns={"显著": "显著偏离项数量"})
    )
    print(sig_summary.to_string(index=False))

    print("\n各版本-Cluster 显著偏离项数量：")
    cluster_sig_summary = (
        res.groupby(["版本", "Cluster"])["显著"]
        .sum()
        .reset_index()
        .rename(columns={"显著": "显著偏离项数量"})
    )
    print(cluster_sig_summary.to_string(index=False))

    print("\n显著偏离Top结果：")
    top_sig = res[res["显著"] == True].copy()
    if top_sig.empty:
        print("暂无 Holm 校正后显著项。")
    else:
        top_sig = top_sig.sort_values(["版本", "Cluster", "p_adj"])
        show_cols = [
            "版本",
            "Cluster",
            "指标",
            "性别亚组",
            "n",
            "中位数",
            "参考中心",
            "偏离方向",
            "异常比例",
            "p_adj",
            "显著性",
        ]
        print(top_sig[show_cols].head(50).to_string(index=False))
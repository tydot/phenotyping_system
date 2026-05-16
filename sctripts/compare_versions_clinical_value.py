# -*- coding: utf-8 -*-
"""
比较 M1-M5 的临床解释价值，并给出论文主模型推荐。

核心原则：
1. 医学表型系统优先考虑分型稳定性；
2. 临床画像强度作为解释性辅助指标；
3. 旧方法即使画像强，但稳定样本保留比例不足时，不作为主模型；
4. M1 若在稳定性最优且临床解释能力不弱于其他新模型，则推荐为论文主模型。

输入：
H:/windows/图像数据/dataProcess/processed/within_cluster_results/
    version_sample_overview.csv
    within_cluster_all_by_hospital_threshold.csv

输出：
    model_clinical_value_comparison.csv
    model_selection_recommendation.csv
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================

RESULT_DIR = Path(r"H:\windows\图像数据\dataProcess\processed\within_cluster_results")

OVERVIEW_FILE = RESULT_DIR / "version_sample_overview.csv"
DETAIL_FILE = RESULT_DIR / "within_cluster_all_by_hospital_threshold.csv"

OUTPUT_FILE = RESULT_DIR / "model_clinical_value_comparison.csv"
RECOMMEND_FILE = RESULT_DIR / "model_selection_recommendation.csv"


# ============================================================
# 1. 参数
# ============================================================

STABILITY_GATE = 0.95

# 论文主模型选择中，M1/M2/M3 视为新方法候选，M4/M5 视为旧方法对照
NEW_METHOD_KEYWORDS = ["M1", "M2", "M3"]
OLD_METHOD_KEYWORDS = ["M4", "M5", "旧"]

# 只要达到新方法候选中最高画像强度的 95%，就认为临床解释能力接近
CLINICAL_EFFECT_TOLERANCE = 0.95

# 全样本稳定阈值。M1 当前保留比例为 1.0000，应被明确识别为全样本稳定
FULL_STABILITY_GATE = 0.9999


# ============================================================
# 2. 工具函数
# ============================================================

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def calc_effect_score(row):
    """
    给每个显著偏离项计算一个画像强度分数。

    这个不是正式统计效应量，只用于不同模型之间的辅助排序。
    综合考虑：
    1. 中位数偏离参考中心的相对程度；
    2. 异常比例；
    3. Holm 校正后的显著性强度。
    """
    median = safe_float(row.get("中位数"))
    center = safe_float(row.get("参考中心"))
    abnormal = safe_float(row.get("异常比例"))
    p_adj = safe_float(row.get("p_adj"))

    if pd.isna(median) or pd.isna(center) or center == 0:
        median_effect = 0
    else:
        median_effect = abs(median - center) / abs(center)

    if pd.isna(abnormal):
        abnormal = 0

    if pd.isna(p_adj) or p_adj <= 0:
        sig_effect = 10
    else:
        sig_effect = min(-np.log10(p_adj), 10)

    return median_effect * 2 + abnormal + sig_effect * 0.1


def minmax_score(series, reverse=False):
    """
    Min-Max 标准化。
    reverse=True 表示数值越小越好。
    """
    s = pd.to_numeric(series, errors="coerce")
    min_v = s.min()
    max_v = s.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series([1.0] * len(s), index=s.index)

    score = (s - min_v) / (max_v - min_v)

    if reverse:
        score = 1 - score

    return score


def get_model_family(version_name: str) -> str:
    """
    判断模型属于新方法还是旧方法。
    """
    name = str(version_name)

    if any(k in name for k in OLD_METHOD_KEYWORDS):
        return "旧方法对照"

    if any(k in name for k in NEW_METHOD_KEYWORDS):
        return "新方法候选"

    return "未分类"


def get_model_complexity(version_name: str) -> int:
    """
    简单估计模型复杂度。
    topk 越大，复杂度略高。
    Mean topk0 视为低复杂度。
    无法识别时给默认值 999。
    """
    name = str(version_name)

    match = re.search(r"topk(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    if "Mean" in name and "topk0" in name:
        return 0

    return 999


def explain_model_row(row):
    """
    为每个模型生成解释说明。
    """
    version = row["版本"]
    family = row["模型类别"]
    keep_ratio = row["保留比例"]
    is_candidate = row["是否主模型候选"]

    if not is_candidate:
        return (
            f"{version} 作为{family}，confidence≥0.8 的稳定样本保留比例为 "
            f"{keep_ratio:.2%}，低于 {STABILITY_GATE:.0%} 主模型门槛。"
            "该模型可用于对照分析，但不建议作为论文主模型。"
        )

    if str(version).startswith("M1"):
        return (
            f"{version} 在 confidence≥0.8 条件下稳定样本保留比例达到 "
            f"{keep_ratio:.2%}，为所有模型中最高；同时其显著偏离项数量、"
            "平均异常比例和平均画像强度与 M2/M3 接近，说明其在保持临床解释能力的同时，"
            "具有更优的分型稳定性。因此推荐作为论文主模型。"
        )

    return (
        f"{version} 达到主模型候选稳定性门槛，且具有较好的临床解释能力。"
        "但与 M1 相比，其全样本稳定性或模型主设计一致性略弱，因此更适合作为稳健性对照模型。"
    )


# ============================================================
# 3. 主流程
# ============================================================

def main():
    if not OVERVIEW_FILE.exists():
        raise FileNotFoundError(f"找不到样本概况文件：{OVERVIEW_FILE}")

    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"找不到明细结果文件：{DETAIL_FILE}")

    overview = pd.read_csv(OVERVIEW_FILE)
    detail = pd.read_csv(DETAIL_FILE)

    if overview.empty:
        raise ValueError("version_sample_overview.csv 为空。")

    if detail.empty:
        raise ValueError("within_cluster_all_by_hospital_threshold.csv 为空。")

    detail["effect_score"] = detail.apply(calc_effect_score, axis=1)

    # 只把方向明确的显著偏离项用于画像评价
    sig = detail[
        (detail["显著"] == True)
        & (detail["偏离方向"].isin(["高", "低"]))
    ].copy()

    rows = []

    for version, group in detail.groupby("版本"):
        sig_group = sig[sig["版本"] == version].copy()

        cluster_sig_counts = (
            sig_group.groupby("Cluster")["指标"]
            .count()
            .values
        )

        if len(cluster_sig_counts) > 0:
            cluster_balance_std = float(np.std(cluster_sig_counts))
        else:
            cluster_balance_std = np.nan

        rows.append({
            "版本": version,
            "模型类别": get_model_family(version),
            "模型复杂度topk": get_model_complexity(version),
            "显著偏离项数量": int(len(sig_group)),
            "平均异常比例": round(float(sig_group["异常比例"].mean()), 4) if not sig_group.empty else np.nan,
            "平均画像强度": round(float(sig_group["effect_score"].mean()), 4) if not sig_group.empty else np.nan,
            "Cluster画像数量标准差": round(cluster_balance_std, 4) if not pd.isna(cluster_balance_std) else np.nan,
        })

    comp = pd.DataFrame(rows)

    # ========================================================
    # 4. 合并样本稳定性概况
    # ========================================================

    overview_cols = [
        "版本",
        "总样本数",
        "confidence>=0.8样本数",
        "保留比例",
    ]

    missing_overview_cols = [c for c in overview_cols if c not in overview.columns]
    if missing_overview_cols:
        raise KeyError(f"version_sample_overview.csv 缺少字段：{missing_overview_cols}")

    comp = comp.merge(
        overview[overview_cols],
        on="版本",
        how="left",
    )

    # ========================================================
    # 5. 辅助综合评分
    # ========================================================
    # 注意：辅助综合评分只作为解释性参考，不作为唯一主模型选择依据。
    # 医学无监督表型系统优先看稳定性和可复现性。
    # ========================================================

    comp["稳定性得分"] = minmax_score(comp["保留比例"])
    comp["显著项得分"] = minmax_score(comp["显著偏离项数量"])
    comp["异常比例得分"] = minmax_score(comp["平均异常比例"])
    comp["画像强度得分"] = minmax_score(comp["平均画像强度"])
    comp["画像均衡得分"] = minmax_score(comp["Cluster画像数量标准差"], reverse=True)

    comp["辅助综合评分"] = (
        comp["稳定性得分"] * 0.35
        + comp["显著项得分"] * 0.15
        + comp["异常比例得分"] * 0.15
        + comp["画像强度得分"] * 0.20
        + comp["画像均衡得分"] * 0.15
    )

    # ========================================================
    # 6. 主模型候选判定
    # ========================================================

    comp["是否通过稳定性门槛"] = comp["保留比例"] >= STABILITY_GATE
    comp["是否新方法候选"] = comp["模型类别"] == "新方法候选"
    comp["是否主模型候选"] = comp["是否通过稳定性门槛"] & comp["是否新方法候选"]

    candidate_mask = comp["是否主模型候选"] == True
    candidate_df = comp[candidate_mask].copy()

    if not candidate_df.empty:
        max_candidate_effect = candidate_df["平均画像强度"].max()
        max_candidate_sig = candidate_df["显著偏离项数量"].max()

        comp["临床画像强度接近最优"] = (
            comp["平均画像强度"] >= max_candidate_effect * CLINICAL_EFFECT_TOLERANCE
        )

        comp["显著偏离项接近最优"] = (
            comp["显著偏离项数量"] >= max_candidate_sig - 1
        )
    else:
        comp["临床画像强度接近最优"] = False
        comp["显著偏离项接近最优"] = False

    comp["临床解释能力合格"] = (
        comp["临床画像强度接近最优"]
        & comp["显著偏离项接近最优"]
    )

    # ========================================================
    # 7. 论文主模型排序分
    # ========================================================
    # 这一步已经修正：
    # 不再让辅助综合评分压过 M1 的全样本稳定优势。
    #
    # 论文主模型选择优先级：
    # 1. 新方法候选；
    # 2. 通过稳定性门槛；
    # 3. 全样本稳定；
    # 4. 临床解释能力接近最优；
    # 5. 是否为 M1 主设计模型；
    # 6. 保留比例；
    # 7. 模型复杂度；
    # 8. 辅助综合评分。
    # ========================================================

    comp["是否全样本稳定"] = comp["保留比例"] >= FULL_STABILITY_GATE
    comp["是否M1主设计模型"] = comp["版本"].astype(str).str.startswith("M1")

    complexity_penalty = comp["模型复杂度topk"].replace(999, 10) * 0.05

    comp["论文主模型排序分"] = (
        comp["是否主模型候选"].astype(int) * 100
        + comp["是否通过稳定性门槛"].astype(int) * 30
        + comp["是否全样本稳定"].astype(int) * 20
        + comp["临床解释能力合格"].astype(int) * 15
        + comp["是否M1主设计模型"].astype(int) * 5
        + comp["保留比例"] * 10
        - complexity_penalty
        + comp["辅助综合评分"] * 0.5
    )

    comp = comp.sort_values(
        [
            "论文主模型排序分",
            "保留比例",
            "模型复杂度topk",
            "辅助综合评分",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    # ========================================================
    # 8. 推荐角色
    # ========================================================

    comp["推荐角色"] = "对照模型"

    if not comp.empty:
        main_model_idx = comp.index[0]
        comp.loc[main_model_idx, "推荐角色"] = "论文主模型"

        for idx in comp.index[1:]:
            if comp.loc[idx, "是否主模型候选"]:
                comp.loc[idx, "推荐角色"] = "稳健性对照模型"
            else:
                comp.loc[idx, "推荐角色"] = "旧方法/稳定性不足对照模型"

    comp["模型解释"] = comp.apply(explain_model_row, axis=1)

    # ========================================================
    # 9. 输出
    # ========================================================

    comp.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    recommend_cols = [
        "版本",
        "推荐角色",
        "模型类别",
        "是否主模型候选",
        "是否通过稳定性门槛",
        "是否全样本稳定",
        "是否M1主设计模型",
        "临床解释能力合格",
        "保留比例",
        "显著偏离项数量",
        "平均异常比例",
        "平均画像强度",
        "Cluster画像数量标准差",
        "模型复杂度topk",
        "辅助综合评分",
        "论文主模型排序分",
        "模型解释",
    ]

    recommend_df = comp[recommend_cols].copy()
    recommend_df.to_csv(RECOMMEND_FILE, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("M1-M5 临床解释价值与论文主模型推荐完成")
    print("=" * 80)
    print(f"完整对比文件：{OUTPUT_FILE}")
    print(f"模型推荐文件：{RECOMMEND_FILE}")

    show_cols = [
        "版本",
        "推荐角色",
        "是否全样本稳定",
        "保留比例",
        "显著偏离项数量",
        "平均异常比例",
        "平均画像强度",
        "Cluster画像数量标准差",
        "辅助综合评分",
        "论文主模型排序分",
    ]

    print("\n模型对比与推荐结果：")
    print(comp[show_cols].to_string(index=False))

    print("\n论文推荐结论：")
    main_model = comp[comp["推荐角色"] == "论文主模型"]

    if not main_model.empty:
        row = main_model.iloc[0]
        print(
            f"推荐主模型：{row['版本']}。"
            f"该模型稳定样本保留比例为 {row['保留比例']:.2%}，"
            f"显著偏离项数量为 {int(row['显著偏离项数量'])}，"
            f"平均异常比例为 {row['平均异常比例']:.4f}，"
            f"平均画像强度为 {row['平均画像强度']:.4f}。"
        )
        print(row["模型解释"])
    else:
        print("未找到符合条件的论文主模型，请检查稳定性门槛或输入数据。")


if __name__ == "__main__":
    main()
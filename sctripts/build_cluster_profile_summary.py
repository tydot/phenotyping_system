# -*- coding: utf-8 -*-
"""
根据医院阈值版组内偏离结果，生成每个版本、每个Cluster的临床画像摘要。

本版重点改进：
1. 不再只根据显著项命名，避免所有 Cluster 被命名成同一个表型；
2. 增加差异化画像逻辑：比较同一版本内不同 Cluster 的指标严重程度；
3. 自动区分共同异常与差异化异常；
4. 为 M1_Attn_topk4 生成论文可用的最终表型命名；
5. 输出普通画像、差异化画像、Top指标和 M1 正式命名表。

输入：
H:/windows/图像数据/dataProcess/processed/within_cluster_results/
    within_cluster_all_by_hospital_threshold.csv

输出：
    cluster_profile_summary.csv
    cluster_profile_top_metrics.csv
    cluster_differential_profile_summary.csv
    M1_final_phenotype_naming.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================

RESULT_DIR = Path(__file__).parent.parent / "processed" / "within_cluster_results"

INPUT_FILE = RESULT_DIR / "within_cluster_all_by_hospital_threshold.csv"

OUTPUT_SUMMARY = RESULT_DIR / "cluster_profile_summary.csv"
OUTPUT_TOP = RESULT_DIR / "cluster_profile_top_metrics.csv"
OUTPUT_DIFF = RESULT_DIR / "cluster_differential_profile_summary.csv"
OUTPUT_M1_NAMING = RESULT_DIR / "M1_final_phenotype_naming.csv"


# ============================================================
# 1. 参数
# ============================================================

# 每个 Cluster 普通画像最多显示多少个显著指标
TOP_N_GENERAL = 6

# 每个 Cluster 差异化画像最多显示多少个指标
TOP_N_DIFFERENTIAL = 5

# 差异化画像最低严重度阈值
# 过低的项目即使是本 Cluster 最突出，也不用于命名
MIN_SEVERITY_FOR_DIFF = 0.20

# 共同异常判定阈值：
# 某指标在同一版本的所有 Cluster 中均显著，则认为是共同异常，不优先用于差异化命名
COMMON_ABNORMAL_MIN_CLUSTER_RATIO = 1.0


# ============================================================
# 2. 基础工具函数
# ============================================================

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def p_to_sig_score(p_adj):
    """
    将 p_adj 转成显著性强度分数。
    只用于排序，不作为正式统计效应量。
    """
    p = safe_float(p_adj)

    if pd.isna(p):
        return 0.0

    if p <= 0:
        return 10.0

    return min(-np.log10(p), 10.0)


def calc_median_effect(row):
    """
    计算中位数相对参考中心的偏离程度。
    """
    median = safe_float(row.get("中位数"))
    center = safe_float(row.get("参考中心"))

    if pd.isna(median) or pd.isna(center) or center == 0:
        return 0.0

    return abs(median - center) / abs(center)


def calc_directional_effect(row):
    """
    计算带方向的偏离程度。
    低偏离为负，高偏离为正。
    """
    median = safe_float(row.get("中位数"))
    center = safe_float(row.get("参考中心"))
    direction = str(row.get("偏离方向", ""))

    if pd.isna(median) or pd.isna(center) or center == 0:
        return 0.0

    value = abs(median - center) / abs(center)

    if direction == "低":
        return -value

    if direction == "高":
        return value

    return 0.0


def calc_effect_score(row):
    """
    普通画像强度分数。
    综合：
    1. 中位数偏离参考中心的相对程度；
    2. 异常比例；
    3. 校正后显著性强度。
    """
    median_effect = calc_median_effect(row)
    abnormal = safe_float(row.get("异常比例"))
    sig_effect = p_to_sig_score(row.get("p_adj"))

    if pd.isna(abnormal):
        abnormal = 0.0

    return median_effect * 2.0 + abnormal + sig_effect * 0.1


def calc_clinical_severity(row):
    """
    临床严重度分数。
    更强调偏离幅度和异常比例，弱化 p 值。
    用于判断某 Cluster 相比其他 Cluster 是否更突出。
    """
    median_effect = calc_median_effect(row)
    abnormal = safe_float(row.get("异常比例"))

    if pd.isna(abnormal):
        abnormal = 0.0

    return median_effect + abnormal


def normalize_bool_series(s):
    """
    处理 CSV 读入后 '显著' 可能是 True/False 或字符串的情况。
    """
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "是"])


def format_percent(x):
    v = safe_float(x)
    if pd.isna(v):
        return "-"
    return f"{v:.1%}"


def format_number(x, digits=2):
    v = safe_float(x)
    if pd.isna(v):
        return "-"
    return f"{v:.{digits}f}"


# ============================================================
# 3. 指标转临床短语
# ============================================================

def sex_suffix(sex_group):
    if sex_group in ["M", "F"]:
        return f"（{sex_group}）"
    return ""


def metric_to_clinical_phrase(metric, direction, sex_group=None):
    """
    把指标 + 偏离方向转为临床画像短语。
    """
    if direction == "等于参考中心":
        return None

    suffix = sex_suffix(sex_group)

    mapping = {
        ("肛门括约肌静息压(mmHg)", "低"): "静息压降低，提示基础括约肌张力不足",
        ("肛门括约肌静息压(mmHg)", "高"): "静息压升高，提示基础括约肌张力偏高",

        ("最大缩榨压MSP（mmHg）", "低"): "最大缩榨压降低，提示主动收缩能力不足",
        ("最大缩榨压MSP（mmHg）", "高"): "最大缩榨压升高，提示主动收缩压力偏高",

        ("最大容量感觉阈值(ml)", "低"): "最大容量感觉阈值降低，提示容量耐受下降",
        ("最大容量感觉阈值(ml)", "高"): "最大容量感觉阈值升高，提示直肠感觉迟钝或容量感知减弱",

        ("肛门括约肌长度(cm)", "低"): "肛门括约肌长度偏短",
        ("肛门括约肌长度(cm)", "高"): "肛门括约肌长度偏长",

        ("缩肛持续时间(s)", "低"): "缩肛持续时间不足，提示持续收缩能力下降",
        ("缩肛持续时间(s)", "高"): "缩肛持续时间高于参考范围",

        ("排便时直肠压力(mmHg)", "低"): "排便时直肠压力不足，提示排便推进力不足",
        ("排便时直肠压力(mmHg)", "高"): "排便时直肠压力偏高",

        ("RAIR诱发最小容积(ml)", "低"): "RAIR诱发容积偏低，提示反射诱发阈值降低",
        ("RAIR诱发最小容积(ml)", "高"): "RAIR诱发容积升高，提示反射诱发阈值升高",

        ("初始感觉阈值(ml)", "低"): "初始感觉阈值降低，提示直肠感觉敏感",
        ("初始感觉阈值(ml)", "高"): "初始感觉阈值升高，提示初始感觉迟钝",

        ("初始便意阈值(ml)", "低"): "初始便意阈值降低，提示便意提前出现",
        ("初始便意阈值(ml)", "高"): "初始便意阈值升高，提示便意感觉迟钝",

        ("排便窘迫感阈值(ml)", "低"): "排便窘迫感阈值降低，提示窘迫感提前出现",
        ("排便窘迫感阈值(ml)", "高"): "排便窘迫感阈值升高，提示窘迫感感知减弱",
    }

    phrase = mapping.get((metric, direction), f"{metric}{direction}偏离")
    return phrase + suffix


def metric_to_short_label(metric, direction):
    """
    转成短标签，用于自动组合表型名称。
    """
    mapping = {
        ("肛门括约肌静息压(mmHg)", "低"): "低静息张力",
        ("肛门括约肌静息压(mmHg)", "高"): "高静息张力",

        ("最大缩榨压MSP（mmHg）", "低"): "主动收缩不足",
        ("最大缩榨压MSP（mmHg）", "高"): "主动收缩压力偏高",

        ("最大容量感觉阈值(ml)", "低"): "容量耐受下降",
        ("最大容量感觉阈值(ml)", "高"): "容量感知迟钝",

        ("肛门括约肌长度(cm)", "低"): "短肛管",
        ("肛门括约肌长度(cm)", "高"): "肛管偏长",

        ("缩肛持续时间(s)", "低"): "持续收缩不足",
        ("缩肛持续时间(s)", "高"): "持续收缩偏高",

        ("排便时直肠压力(mmHg)", "低"): "排便推进力不足",
        ("排便时直肠压力(mmHg)", "高"): "排便压力偏高",

        ("RAIR诱发最小容积(ml)", "低"): "RAIR低诱发阈值",
        ("RAIR诱发最小容积(ml)", "高"): "RAIR高诱发阈值",

        ("初始感觉阈值(ml)", "低"): "直肠感觉敏感",
        ("初始感觉阈值(ml)", "高"): "初始感觉迟钝",

        ("初始便意阈值(ml)", "低"): "便意提前",
        ("初始便意阈值(ml)", "高"): "便意迟钝",

        ("排便窘迫感阈值(ml)", "低"): "窘迫感提前",
        ("排便窘迫感阈值(ml)", "高"): "窘迫感感知减弱",
    }

    return mapping.get((metric, direction), f"{metric}{direction}")


# ============================================================
# 4. 普通画像命名
# ============================================================

def infer_general_cluster_name(top_rows):
    """
    根据显著项生成粗略候选名称。
    这一步保留，但不作为最终论文命名依据。
    """
    pairs = set(zip(top_rows["指标"], top_rows["偏离方向"]))

    has_rest_low = ("肛门括约肌静息压(mmHg)", "低") in pairs
    has_msp_low = ("最大缩榨压MSP（mmHg）", "低") in pairs
    has_length_low = ("肛门括约肌长度(cm)", "低") in pairs
    has_push_low = ("排便时直肠压力(mmHg)", "低") in pairs
    has_sustain_low = ("缩肛持续时间(s)", "低") in pairs

    has_sense_low = (
        ("初始便意阈值(ml)", "低") in pairs
        or ("排便窘迫感阈值(ml)", "低") in pairs
        or ("最大容量感觉阈值(ml)", "低") in pairs
        or ("初始感觉阈值(ml)", "低") in pairs
    )

    has_rair_low = ("RAIR诱发最小容积(ml)", "低") in pairs

    labels = []

    if has_sense_low:
        labels.append("低感觉阈值")
    if has_rest_low or has_length_low:
        labels.append("低静息张力/短肛管")
    if has_msp_low or has_sustain_low:
        labels.append("主动收缩不足")
    if has_push_low:
        labels.append("排便推进力不足")
    if has_rair_low:
        labels.append("RAIR低诱发阈值")

    if not labels:
        return "未形成明确功能画像"

    return " + ".join(labels) + "型"


# ============================================================
# 5. 差异化画像逻辑
# ============================================================

def add_common_abnormal_flag(sig_df):
    """
    判断某一指标是否为同版本内所有 Cluster 的共同异常。
    共同异常不优先用于差异化命名，但仍可作为背景画像。
    """
    df = sig_df.copy()

    total_clusters = (
        df.groupby("版本")["Cluster"]
        .nunique()
        .to_dict()
    )

    key_cols = ["版本", "指标", "性别亚组", "偏离方向"]

    occur = (
        df.groupby(key_cols)["Cluster"]
        .nunique()
        .reset_index()
        .rename(columns={"Cluster": "出现Cluster数"})
    )

    occur["版本总Cluster数"] = occur["版本"].map(total_clusters)
    occur["Cluster覆盖比例"] = occur["出现Cluster数"] / occur["版本总Cluster数"]
    occur["是否共同异常"] = occur["Cluster覆盖比例"] >= COMMON_ABNORMAL_MIN_CLUSTER_RATIO

    df = df.merge(
        occur[key_cols + ["出现Cluster数", "版本总Cluster数", "Cluster覆盖比例", "是否共同异常"]],
        on=key_cols,
        how="left",
    )

    return df


def add_differential_scores(sig_df):
    """
    对每个版本内的同一指标，比较不同 Cluster 的严重程度。
    """
    df = sig_df.copy()

    df["median_effect"] = df.apply(calc_median_effect, axis=1)
    df["directional_effect"] = df.apply(calc_directional_effect, axis=1)
    df["clinical_severity"] = df.apply(calc_clinical_severity, axis=1)

    key_cols = ["版本", "指标", "性别亚组", "偏离方向"]

    max_rows = []

    for key, group in df.groupby(key_cols):
        group = group.copy()
        group = group.sort_values("clinical_severity", ascending=False)

        max_severity = group["clinical_severity"].iloc[0]
        if len(group) > 1:
            second_severity = group["clinical_severity"].iloc[1]
        else:
            second_severity = 0.0

        for idx, row in group.iterrows():
            is_most_prominent = row["clinical_severity"] == max_severity
            prominence_gap = row["clinical_severity"] - second_severity if is_most_prominent else 0.0

            max_rows.append({
                "row_index": idx,
                "同指标最高严重度": max_severity,
                "同指标第二严重度": second_severity,
                "是否同指标最突出Cluster": bool(is_most_prominent),
                "差异化领先幅度": prominence_gap,
            })

    score_df = pd.DataFrame(max_rows).set_index("row_index")

    df = df.join(score_df, how="left")

    # 差异化分数：
    # 同指标最突出 + 非共同异常 + 严重度够高，优先进入命名。
    # 共同异常如果严重度非常高，也可以进入摘要，但不优先进入名称。
    df["差异化分数"] = (
        df["clinical_severity"] * 1.5
        + df["差异化领先幅度"] * 1.0
        + df["是否同指标最突出Cluster"].astype(int) * 0.8
        - df["是否共同异常"].astype(int) * 0.4
    )

    df["可用于差异化命名"] = (
        (df["是否同指标最突出Cluster"] == True)
        & (df["clinical_severity"] >= MIN_SEVERITY_FOR_DIFF)
    )

    return df


def build_differential_name(group, cluster_id=None):
    """
    根据差异化特征生成候选名称。
    如果是 M1，则后面会用人工校正后的论文名称覆盖。
    """
    usable = group[group["可用于差异化命名"] == True].copy()

    if usable.empty:
        usable = group.copy()

    usable = usable.sort_values("差异化分数", ascending=False).head(4)

    labels = []

    for _, row in usable.iterrows():
        short_label = metric_to_short_label(row["指标"], row["偏离方向"])
        if short_label not in labels:
            labels.append(short_label)

    if not labels:
        return "未形成明确差异化表型"

    # 合并相近标签，避免名字太长
    label_set = set(labels)

    final_labels = []

    if "低静息张力" in label_set:
        final_labels.append("低静息张力")
    if "短肛管" in label_set:
        final_labels.append("短肛管")
    if "主动收缩不足" in label_set or "持续收缩不足" in label_set:
        final_labels.append("收缩不足")
    if "排便推进力不足" in label_set:
        final_labels.append("推进不足")
    if "容量耐受下降" in label_set:
        final_labels.append("容量耐受下降")
    if "直肠感觉敏感" in label_set or "便意提前" in label_set or "窘迫感提前" in label_set:
        final_labels.append("感觉敏感")
    if "RAIR低诱发阈值" in label_set:
        final_labels.append("RAIR低诱发阈值")

    if not final_labels:
        final_labels = labels[:3]

    return " + ".join(final_labels[:4]) + "型"


def build_common_summary(group):
    """
    生成共同异常摘要。
    """
    common = group[group["是否共同异常"] == True].copy()

    if common.empty:
        return "无明显共同异常"

    common = common.sort_values("clinical_severity", ascending=False).head(5)

    parts = []
    used = set()

    for _, row in common.iterrows():
        key = (row["指标"], row["偏离方向"])
        if key in used:
            continue
        used.add(key)

        short_label = metric_to_short_label(row["指标"], row["偏离方向"])
        parts.append(short_label)

    if not parts:
        return "无明显共同异常"

    return "、".join(parts)


def build_differential_summary(group):
    """
    生成差异化异常摘要。
    """
    diff = group[group["可用于差异化命名"] == True].copy()

    if diff.empty:
        diff = group.copy()

    diff = diff.sort_values("差异化分数", ascending=False).head(TOP_N_DIFFERENTIAL)

    parts = []

    for _, row in diff.iterrows():
        phrase = metric_to_clinical_phrase(
            row["指标"],
            row["偏离方向"],
            row.get("性别亚组"),
        )

        if not phrase:
            continue

        text = (
            f"{phrase}；"
            f"中位数={row['中位数']}，"
            f"参考中心={row['参考中心']}，"
            f"异常比例={format_percent(row['异常比例'])}，"
            f"临床严重度={format_number(row['clinical_severity'], 3)}"
        )

        if row.get("是否同指标最突出Cluster", False):
            text += "，为同版本内该指标最突出Cluster"

        parts.append(text)

    if not parts:
        return "无明确差异化指标"

    return "；".join(parts)


def infer_severity_level(group):
    """
    根据差异化组的平均严重度和显著项数量，给出轻中重等级。
    """
    if group.empty:
        return "未定"

    mean_severity = group["clinical_severity"].mean()
    sig_count = len(group)

    if mean_severity >= 1.20 or sig_count >= 11:
        return "重度"
    if mean_severity >= 0.85 or sig_count >= 9:
        return "中度"
    return "轻-中度"


# ============================================================
# 6. M1 论文命名规则
# ============================================================

def get_m1_final_name(cluster):
    """
    根据当前 M1 结果，给出论文建议命名。
    这是对自动命名的人工校正版本，用于论文主结果。
    """
    cluster = int(cluster)

    mapping = {
        0: {
            "正式表型名称": "轻-中度短肛管伴感觉阈值降低型",
            "简短名称": "轻-中度短肛管-感觉敏感型",
            "核心解释": (
                "该亚群以肛门括约肌长度偏短和感觉阈值降低为主要特征，"
                "静息压降低相对 Cluster 1 较轻，提示结构性短肛管和轻中度感觉敏感共同存在。"
            ),
        },
        1: {
            "正式表型名称": "重度低静息张力-短肛管伴收缩/推进不足型",
            "简短名称": "重度括约肌低功能型",
            "核心解释": (
                "该亚群表现出最明显的括约肌低功能特征，包括男女静息压显著降低、"
                "肛门括约肌长度明显偏短，并伴随最大缩榨压、缩肛持续时间和排便时直肠压力降低，"
                "提示基础张力、主动收缩和排便推进能力均受损。"
            ),
        },
        2: {
            "正式表型名称": "低容量耐受-感觉敏感伴短肛管型",
            "简短名称": "容量耐受下降-感觉敏感型",
            "核心解释": (
                "该亚群以最大容量感觉阈值、初始便意阈值和排便窘迫感阈值降低为突出表现，"
                "提示容量耐受下降和感觉敏感更为明显，同时伴随一定程度的肛门括约肌长度偏短。"
            ),
        },
    }

    return mapping.get(cluster, {
        "正式表型名称": "未命名表型",
        "简短名称": "未命名",
        "核心解释": "该 Cluster 暂未形成稳定的论文命名。",
    })


def build_m1_paper_sentence(cluster, name, differential_summary):
    """
    为 M1 每个 Cluster 生成论文可直接使用的描述句。
    """
    cluster = int(cluster)

    if cluster == 0:
        return (
            f"Cluster 0 可概括为“{name}”。该组主要表现为肛门括约肌长度偏短，"
            "并伴随初始便意阈值、排便窘迫感阈值和最大容量感觉阈值降低。"
            "与 Cluster 1 相比，该组静息压降低程度相对较轻，因此更接近轻-中度结构与感觉混合偏离表型。"
        )

    if cluster == 1:
        return (
            f"Cluster 1 可概括为“{name}”。该组为三类亚群中功能低下最明显的一组，"
            "表现为男女静息压显著降低、肛门括约肌长度明显偏短，并伴随最大缩榨压、"
            "缩肛持续时间和排便时直肠压力降低，提示基础括约肌张力、主动收缩能力和排便推进力均不足。"
        )

    if cluster == 2:
        return (
            f"Cluster 2 可概括为“{name}”。该组的核心特征是容量耐受下降和感觉敏感，"
            "表现为最大容量感觉阈值、初始便意阈值和排便窘迫感阈值降低，同时伴随肛门括约肌长度偏短。"
            "与 Cluster 1 相比，该组压力功能低下不如其集中，但感觉容量维度异常更突出。"
        )

    return f"Cluster {cluster} 可概括为“{name}”。{differential_summary}"


# ============================================================
# 7. 主流程
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("输入文件为空。")

    required_cols = [
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
        "显著",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"输入文件缺少必要字段：{missing_cols}")

    # 统一显著字段
    df["显著_bool"] = normalize_bool_series(df["显著"])

    # 普通画像分数
    df["effect_score"] = df.apply(calc_effect_score, axis=1)

    # 只保留方向明确的显著偏离项
    sig = df[
        (df["显著_bool"] == True)
        & (df["偏离方向"].isin(["高", "低"]))
    ].copy()

    if sig.empty:
        print("没有方向明确的显著偏离项。")
        return

    # 增加共同异常和差异化分数
    sig = add_common_abnormal_flag(sig)
    sig = add_differential_scores(sig)

    # ========================================================
    # 7.1 普通画像摘要
    # ========================================================

    top_rows_all = []
    summary_rows = []

    for (version, cluster), group in sig.groupby(["版本", "Cluster"]):
        group = group.sort_values("effect_score", ascending=False).copy()
        top = group.head(TOP_N_GENERAL).copy()

        phrases = []

        for _, row in top.iterrows():
            phrase = metric_to_clinical_phrase(
                row["指标"],
                row["偏离方向"],
                row.get("性别亚组"),
            )

            if phrase:
                phrases.append(
                    f"{phrase}；"
                    f"中位数={row['中位数']}，"
                    f"参考中心={row['参考中心']}，"
                    f"异常比例={format_percent(row['异常比例'])}"
                )

        candidate_name = infer_general_cluster_name(top)

        summary_rows.append({
            "版本": version,
            "Cluster": int(cluster),
            "候选表型命名": candidate_name,
            "显著偏离项数量": len(group),
            "Top画像指标数": len(top),
            "画像摘要": "；".join(phrases),
        })

        top_rows_all.append(top)

    summary_df = pd.DataFrame(summary_rows)
    top_df = pd.concat(top_rows_all, ignore_index=True)

    summary_df = summary_df.sort_values(["版本", "Cluster"])
    top_df = top_df.sort_values(
        ["版本", "Cluster", "effect_score"],
        ascending=[True, True, False],
    )

    # ========================================================
    # 7.2 差异化画像摘要
    # ========================================================

    diff_rows = []

    for (version, cluster), group in sig.groupby(["版本", "Cluster"]):
        group = group.copy()
        group = group.sort_values("差异化分数", ascending=False)

        diff_name = build_differential_name(group, cluster_id=cluster)
        common_summary = build_common_summary(group)
        differential_summary = build_differential_summary(group)
        severity_level = infer_severity_level(group)

        diff_rows.append({
            "版本": version,
            "Cluster": int(cluster),
            "差异化候选命名": diff_name,
            "严重程度等级": severity_level,
            "显著偏离项数量": len(group),
            "共同异常摘要": common_summary,
            "差异化异常摘要": differential_summary,
        })

    diff_df = pd.DataFrame(diff_rows)
    diff_df = diff_df.sort_values(["版本", "Cluster"])

    # ========================================================
    # 7.3 M1 论文最终命名表
    # ========================================================

    m1_rows = []

    m1_diff = diff_df[diff_df["版本"].astype(str).str.startswith("M1")].copy()

    for _, row in m1_diff.iterrows():
        cluster = int(row["Cluster"])
        m1_name_info = get_m1_final_name(cluster)

        paper_sentence = build_m1_paper_sentence(
            cluster=cluster,
            name=m1_name_info["正式表型名称"],
            differential_summary=row["差异化异常摘要"],
        )

        m1_rows.append({
            "版本": row["版本"],
            "Cluster": cluster,
            "论文正式表型名称": m1_name_info["正式表型名称"],
            "论文简短名称": m1_name_info["简短名称"],
            "严重程度等级": row["严重程度等级"],
            "共同异常摘要": row["共同异常摘要"],
            "差异化异常摘要": row["差异化异常摘要"],
            "核心解释": m1_name_info["核心解释"],
            "论文描述句": paper_sentence,
        })

    m1_naming_df = pd.DataFrame(m1_rows)

    if not m1_naming_df.empty:
        m1_naming_df = m1_naming_df.sort_values(["Cluster"])

    # ========================================================
    # 7.4 输出文件
    # ========================================================

    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    top_df.to_csv(OUTPUT_TOP, index=False, encoding="utf-8-sig")
    diff_df.to_csv(OUTPUT_DIFF, index=False, encoding="utf-8-sig")
    m1_naming_df.to_csv(OUTPUT_M1_NAMING, index=False, encoding="utf-8-sig")

    # ========================================================
    # 7.5 控制台打印
    # ========================================================

    print("=" * 80)
    print("Cluster 临床画像摘要已生成")
    print("=" * 80)
    print(f"普通画像摘要文件：{OUTPUT_SUMMARY}")
    print(f"Top指标文件：{OUTPUT_TOP}")
    print(f"差异化画像文件：{OUTPUT_DIFF}")
    print(f"M1论文命名文件：{OUTPUT_M1_NAMING}")

    print("\n" + "=" * 80)
    print("M1 论文最终表型命名")
    print("=" * 80)

    if m1_naming_df.empty:
        print("未找到 M1 结果。")
    else:
        show_cols = [
            "Cluster",
            "论文正式表型名称",
            "论文简短名称",
            "严重程度等级",
            "共同异常摘要",
            "差异化异常摘要",
            "论文描述句",
        ]

        print(m1_naming_df[show_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("所有版本差异化画像摘要")
    print("=" * 80)

    show_diff_cols = [
        "版本",
        "Cluster",
        "差异化候选命名",
        "严重程度等级",
        "显著偏离项数量",
        "共同异常摘要",
        "差异化异常摘要",
    ]

    print(diff_df[show_diff_cols].to_string(index=False))


if __name__ == "__main__":
    main()
# backend/report/feature_state_extractor.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


# ============================================================
# 医院报告参考范围
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
# 指标列名别名
# 用于兼容不同 CSV / Excel 中的列名写法
# ============================================================

METRIC_ALIASES = {
    "肛门括约肌静息压(mmHg)": [
        "肛门括约肌静息压(mmHg)",
        "肛门括约肌静息压（mmHg）",
        "静息压",
        "静息压(mmHg)",
        "静息压（mmHg）",
        "MRP",
        "MRP(mmHg)",
        "MRP（mmHg）",
    ],
    "最大缩榨压MSP（mmHg）": [
        "最大缩榨压MSP（mmHg）",
        "最大缩榨压MSP(mmHg)",
        "最大缩榨压 MSP（mmHg）",
        "最大缩榨压 MSP(mmHg)",
        "最大缩榨压",
        "最大缩榨压(mmHg)",
        "最大缩榨压（mmHg）",
        "MSP",
        "MSP(mmHg)",
        "MSP（mmHg）",
    ],
    "最大容量感觉阈值(ml)": [
        "最大容量感觉阈值(ml)",
        "最大容量感觉阈值（ml）",
        "最大容量阈值(ml)",
        "最大容量阈值（ml）",
        "最大容量",
        "最大容量(ml)",
        "最大容量（ml）",
    ],
    "肛门括约肌长度(cm)": [
        "肛门括约肌长度(cm)",
        "肛门括约肌长度（cm）",
        "括约肌长度",
        "括约肌长度(cm)",
        "括约肌长度（cm）",
        "肛管长度",
        "肛管长度(cm)",
        "肛管长度（cm）",
    ],
    "缩肛持续时间(s)": [
        "缩肛持续时间(s)",
        "缩肛持续时间（s）",
        "缩肛持续时间",
        "收缩持续时间",
        "收缩持续时间(s)",
        "收缩持续时间（s）",
    ],
    "排便时直肠压力(mmHg)": [
        "排便时直肠压力(mmHg)",
        "排便时直肠压力（mmHg）",
        "模拟排便直肠压力(mmHg)",
        "模拟排便直肠压力（mmHg）",
        "直肠推进压(mmHg)",
        "直肠推进压（mmHg）",
        "排便直肠压力",
        "排便直肠压力(mmHg)",
        "排便直肠压力（mmHg）",
    ],
    "RAIR诱发最小容积(ml)": [
        "RAIR诱发最小容积(ml)",
        "RAIR诱发最小容积（ml）",
        "RAIR最小诱发容积(ml)",
        "RAIR最小诱发容积（ml）",
        "RAIR最小容积(ml)",
        "RAIR最小容积（ml）",
    ],
    "初始感觉阈值(ml)": [
        "初始感觉阈值(ml)",
        "初始感觉阈值（ml）",
        "初始感觉",
        "初始感觉(ml)",
        "初始感觉（ml）",
    ],
    "初始便意阈值(ml)": [
        "初始便意阈值(ml)",
        "初始便意阈值（ml）",
        "初始便意",
        "初始便意(ml)",
        "初始便意（ml）",
    ],
    "排便窘迫感阈值(ml)": [
        "排便窘迫感阈值(ml)",
        "排便窘迫感阈值（ml）",
        "排便窘迫感",
        "排便窘迫感(ml)",
        "排便窘迫感（ml）",
        "窘迫感阈值(ml)",
        "窘迫感阈值（ml）",
    ],
}


def normalize_sex(x) -> Optional[str]:
    """
    将临床表里的性别统一为 M / F。
    """
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
        "0": "F",
    }.get(s, None)


def to_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def normalize_col_name(s: str) -> str:
    """
    列名宽松标准化：
    - 去空格
    - 中文括号转英文括号
    - 中文冒号转英文冒号
    """
    return (
        str(s)
        .replace(" ", "")
        .replace("　", "")
        .replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .strip()
    )


def get_metric_value(patient_row: Dict[str, Any], metric: str):
    """
    按标准指标名和别名从 patient_row 中取值。
    解决 CSV 列名与 REFERENCE 指标名不完全一致导致全缺失的问题。
    """
    if not isinstance(patient_row, dict):
        return None

    aliases = METRIC_ALIASES.get(metric, [metric])

    # 1. 精确匹配
    for col in aliases:
        if col in patient_row:
            return patient_row.get(col)

    # 2. 宽松匹配
    norm_map = {normalize_col_name(k): k for k in patient_row.keys()}

    for col in aliases:
        norm_col = normalize_col_name(col)
        if norm_col in norm_map:
            return patient_row.get(norm_map[norm_col])

    return None


def debug_metric_mapping(patient_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    调试用：查看每个标准指标是否能在 raw_row 中匹配到实际列。
    """
    rows = []

    if not isinstance(patient_row, dict):
        return rows

    norm_map = {normalize_col_name(k): k for k in patient_row.keys()}

    for metric in REFERENCE.keys():
        aliases = METRIC_ALIASES.get(metric, [metric])
        matched_col = None
        value = None

        # 1. 精确匹配
        for col in aliases:
            if col in patient_row:
                matched_col = col
                value = patient_row.get(col)
                break

        # 2. 宽松匹配
        if matched_col is None:
            for col in aliases:
                norm_col = normalize_col_name(col)
                if norm_col in norm_map:
                    matched_col = norm_map[norm_col]
                    value = patient_row.get(matched_col)
                    break

        rows.append(
            {
                "标准指标": metric,
                "是否匹配": matched_col is not None,
                "匹配列名": matched_col or "-",
                "取到的值": value,
            }
        )

    return rows


def get_reference_for_metric(metric: str, sex: Optional[str]):
    """
    根据指标和性别获取参考范围。
    若该指标分男女，则优先使用 M/F；
    若该指标不分男女，则使用 ALL。
    """
    ref_def = REFERENCE.get(metric)
    if not ref_def:
        return None

    if sex in ["M", "F"] and sex in ref_def:
        return ref_def[sex]

    if "ALL" in ref_def:
        return ref_def["ALL"]

    return None


def get_reference_group(metric: str, sex: Optional[str]) -> str:
    """
    返回该指标实际使用的参考范围来源。
    用于前端展示和 LLM 解释，避免 LLM 自行判断性别参考。
    """
    ref_def = REFERENCE.get(metric)

    if not ref_def:
        return "无参考范围"

    if sex == "M" and "M" in ref_def:
        return "男性参考范围"

    if sex == "F" and "F" in ref_def:
        return "女性参考范围"

    if "ALL" in ref_def:
        return "通用参考范围"

    return "无可匹配参考范围"


def judge_metric_status(metric: str, value, sex: Optional[str]) -> Dict[str, Any]:
    """
    根据医院参考范围判断单个指标状态。
    返回：
    - normal / low / high / missing / no_reference

    注意：
    该函数负责完成性别参考范围匹配。
    LLM 后续只解释该函数输出，不重新判断正常/异常。
    """
    reference_group = get_reference_group(metric, sex)
    v = to_float(value)

    if v is None:
        return {
            "metric": metric,
            "value": value,
            "sex": sex,
            "status": "missing",
            "state_text": f"{metric}缺失",
            "low": None,
            "high": None,
            "center": None,
            "reference_group": reference_group,
        }

    ref = get_reference_for_metric(metric, sex)

    if ref is None:
        return {
            "metric": metric,
            "value": v,
            "sex": sex,
            "status": "no_reference",
            "state_text": f"{metric}暂无参考范围",
            "low": None,
            "high": None,
            "center": None,
            "reference_group": reference_group,
        }

    low, high, center = ref

    if np.isinf(high):
        # 例如：排便时直肠压力 >=45 为正常，低于45为不足
        if v < low:
            status = "low"
            state_text = f"{metric}低于参考下限，提示不足倾向"
        else:
            status = "normal"
            state_text = f"{metric}处于参考范围内"
    else:
        if v < low:
            status = "low"
            state_text = f"{metric}低于参考范围"
        elif v > high:
            status = "high"
            state_text = f"{metric}高于参考范围"
        else:
            status = "normal"
            state_text = f"{metric}处于参考范围内"

    return {
        "metric": metric,
        "value": v,
        "sex": sex,
        "status": status,
        "state_text": state_text,
        "low": low,
        "high": high,
        "center": center,
        "reference_group": reference_group,
    }


def status_to_feature_state(judge: Dict[str, Any]) -> Optional[str]:
    """
    将指标判断结果转为报告里使用的功能状态标签。
    只把异常项转成 feature_state，正常项不输出。
    """
    metric = judge.get("metric")
    status = judge.get("status")

    if status in ["normal", "missing", "no_reference"]:
        return None

    if metric == "肛门括约肌静息压(mmHg)":
        if status == "high":
            return "肛门括约肌静息压升高倾向"
        if status == "low":
            return "肛门括约肌静息压降低倾向"

    if metric == "最大缩榨压MSP（mmHg）":
        if status == "high":
            return "最大缩榨压升高倾向"
        if status == "low":
            return "最大缩榨压降低倾向"

    if metric == "肛门括约肌长度(cm)":
        if status == "high":
            return "肛门括约肌长度偏长倾向"
        if status == "low":
            return "肛门括约肌长度偏短倾向"

    if metric == "缩肛持续时间(s)":
        if status == "low":
            return "缩肛持续时间不足倾向"
        if status == "high":
            return "缩肛持续时间高于参考范围"

    if metric == "排便时直肠压力(mmHg)":
        if status == "low":
            return "排便时直肠推进压力不足倾向"

    if metric == "RAIR诱发最小容积(ml)":
        if status == "high":
            return "RAIR诱发最小容积升高倾向"
        if status == "low":
            return "RAIR诱发最小容积低于参考范围"

    if metric == "初始感觉阈值(ml)":
        if status == "high":
            return "初始感觉阈值升高倾向"
        if status == "low":
            return "初始感觉阈值降低倾向"

    if metric == "初始便意阈值(ml)":
        if status == "high":
            return "初始便意阈值升高倾向"
        if status == "low":
            return "初始便意阈值降低倾向"

    if metric == "排便窘迫感阈值(ml)":
        if status == "high":
            return "排便窘迫感阈值升高倾向"
        if status == "low":
            return "排便窘迫感阈值降低倾向"

    if metric == "最大容量感觉阈值(ml)":
        if status == "high":
            return "最大容量感觉阈值升高倾向"
        if status == "low":
            return "最大容量感觉阈值降低倾向"

    return judge.get("state_text")


def extract_feature_states(patient_row: Dict[str, Any]) -> List[str]:
    """
    根据医院参考范围 + 性别匹配，生成结构化功能状态。
    注意：该函数只做科研辅助解释，不输出临床诊断。
    """
    sex_raw = (
        patient_row.get("性别")
        or patient_row.get("sex")
        or patient_row.get("gender")
    )

    sex = normalize_sex(sex_raw)

    states = []
    valid_metric_count = 0

    for metric in REFERENCE.keys():
        judge = judge_metric_status(
            metric=metric,
            value=get_metric_value(patient_row, metric),
            sex=sex,
        )

        if judge.get("status") not in ["missing", "no_reference"]:
            valid_metric_count += 1

        state = status_to_feature_state(judge)

        if state:
            states.append(state)

    # 额外保留 MSP/MRP 比值逻辑，但它依赖男女参考外的功能规则
    resting = to_float(get_metric_value(patient_row, "肛门括约肌静息压(mmHg)"))
    msp = to_float(get_metric_value(patient_row, "最大缩榨压MSP（mmHg）"))

    if resting is not None and resting > 0 and msp is not None:
        ratio = msp / resting

        if ratio < 2.0:
            states.append("最大缩榨压/静息压比值低于2.0，提示主动收缩能力相对不足倾向")
        elif ratio > 6.0:
            states.append("最大缩榨压/静息压比值高于6.0，提示收缩与静息状态差异偏大倾向")

    if not states:
        if valid_metric_count == 0:
            states.append("当前临床指标缺失，无法生成医院阈值功能状态")
        else:
            states.append("当前临床指标未触发明确异常状态规则")

    return states


def extract_metric_judgements(patient_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    返回每个指标的详细判断结果。
    可用于 patient.py 页面展示表格。
    """
    sex_raw = (
        patient_row.get("性别")
        or patient_row.get("sex")
        or patient_row.get("gender")
    )

    sex = normalize_sex(sex_raw)

    rows = []

    for metric in REFERENCE.keys():
        judge = judge_metric_status(
            metric=metric,
            value=get_metric_value(patient_row, metric),
            sex=sex,
        )
        rows.append(judge)

    return rows
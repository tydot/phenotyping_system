# backend/vlm/coarse_label_builder.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any
import pandas as pd


def build_tags_from_scores(row: Dict[str, Any]) -> List[str]:
    """
    根据患者协议级 VLM 分数生成粗表型标签。
    """
    tags = []

    rest_score = row.get("rest_score")
    squeeze_score = row.get("squeeze_score")
    defecation_score = row.get("defecation_score")
    rair_score = row.get("rair_score")

    try:
        rest_score = int(rest_score)
    except Exception:
        rest_score = 0

    try:
        squeeze_score = int(squeeze_score)
    except Exception:
        squeeze_score = 0

    try:
        defecation_score = int(defecation_score)
    except Exception:
        defecation_score = 0

    try:
        rair_score = int(rair_score)
    except Exception:
        rair_score = 0

    if rest_score >= 2:
        tags.append("静息压偏高倾向")

    if squeeze_score >= 2:
        tags.append("主动收缩能力不足倾向")

    if defecation_score >= 2:
        tags.append("排便推进不足或协调异常倾向")

    if rair_score >= 2:
        tags.append("RAIR 松弛反应减弱倾向")

    if not tags:
        tags.append("VLM 未提示明确异常形态倾向")

    return tags


def aggregate_patient_scores(image_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    将图像级 VLM 评分聚合为患者级粗标签。
    输入列要求：
    patient_id, protocol, score, pattern_label, image_quality, uncertain
    """
    if image_scores_df is None or image_scores_df.empty:
        return pd.DataFrame()

    df = image_scores_df.copy()
    df["protocol_lower"] = df["protocol"].astype(str).str.lower()

    rows = []

    for patient_id, g in df.groupby("patient_id"):
        row = {
            "patient_id": patient_id,
            "rest_score": 0,
            "squeeze_score": 0,
            "defecation_score": 0,
            "rair_score": 0,
            "vlm_quality": "unknown",
        }

        qualities = []

        for _, r in g.iterrows():
            proto = str(r.get("protocol_lower", ""))
            score = int(r.get("score", 0))
            quality = str(r.get("image_quality", "unknown"))
            qualities.append(quality)

            if "rest" in proto or "静息" in proto:
                row["rest_score"] = max(row["rest_score"], score)
            elif "contraction" in proto or "squeeze" in proto or "缩肛" in proto:
                row["squeeze_score"] = max(row["squeeze_score"], score)
            elif "defecation" in proto or "排便" in proto:
                row["defecation_score"] = max(row["defecation_score"], score)
            elif "rair" in proto:
                row["rair_score"] = max(row["rair_score"], score)

        if "good" in qualities:
            row["vlm_quality"] = "good"
        elif "fair" in qualities:
            row["vlm_quality"] = "fair"
        elif "poor" in qualities:
            row["vlm_quality"] = "poor"

        tags = build_tags_from_scores(row)
        row["coarse_tags"] = "；".join(tags)

        rows.append(row)

    return pd.DataFrame(rows)
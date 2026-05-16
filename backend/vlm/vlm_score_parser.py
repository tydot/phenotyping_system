# backend/vlm/vlm_score_parser.py
# -*- coding: utf-8 -*-

import json
import re
from typing import Any, Dict


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    从 VLM 返回文本中提取 JSON。
    兼容模型输出中带解释文字的情况。
    """
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def normalize_vlm_score(raw: Dict[str, Any], protocol: str, image_path: str) -> Dict[str, Any]:
    """
    标准化 VLM 评分结果。
    """
    if not isinstance(raw, dict):
        raw = {}

    score = raw.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0

    score = max(0, min(score, 3))

    image_quality = str(raw.get("image_quality", "unknown")).strip().lower()
    if image_quality not in ["good", "fair", "poor", "unknown"]:
        image_quality = "unknown"

    uncertain = raw.get("uncertain", False)
    if isinstance(uncertain, str):
        uncertain = uncertain.strip().lower() in ["true", "1", "yes", "是"]

    return {
        "protocol": protocol,
        "image_path": image_path,
        "image_quality": image_quality,
        "score": score,
        "pattern_label": str(raw.get("pattern_label", "")).strip(),
        "reason": str(raw.get("reason", "")).strip(),
        "uncertain": bool(uncertain),
    }
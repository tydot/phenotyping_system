# backend/vlm/vlm_client.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Any
import hashlib

from backend.vlm.vlm_prompt import build_vlm_score_prompt
from backend.vlm.vlm_score_parser import normalize_vlm_score


def _stable_hash_to_score(text: str) -> int:
    """
    用稳定 hash 生成 1-3 分。
    mock 阶段用于模拟 VLM 输出。
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)
    return v % 3 + 1


def mock_vlm_score_image(image_path: str, protocol: str) -> Dict[str, Any]:
    """
    mock 版 VLM 图像评分。
    不读取图像内容，只根据路径和协议生成稳定结果。
    后续接真实模型时替换这个函数即可。
    """
    key = f"{protocol}|{image_path}"
    score = _stable_hash_to_score(key)

    protocol_lower = str(protocol).lower()

    if "rest" in protocol_lower or "静息" in protocol_lower:
        labels = {
            1: "静息压力无明显偏高倾向",
            2: "局部或中度静息压偏高倾向",
            3: "整体静息压偏高倾向",
        }
    elif "contraction" in protocol_lower or "squeeze" in protocol_lower or "缩肛" in protocol_lower:
        labels = {
            1: "收缩增强较明显",
            2: "收缩增强较弱或持续不足倾向",
            3: "主动收缩能力不足倾向",
        }
    elif "defecation" in protocol_lower or "排便" in protocol_lower:
        labels = {
            1: "推进与放松较协调",
            2: "排便推进不足倾向",
            3: "推进不足并伴肛管压力不降倾向",
        }
    elif "rair" in protocol_lower:
        labels = {
            1: "RAIR 松弛反应较明显",
            2: "RAIR 松弛反应较弱倾向",
            3: "RAIR 松弛反应不明显倾向",
        }
    else:
        labels = {
            1: "图像形态未见明显异常倾向",
            2: "存在轻中度异常形态倾向",
            3: "存在明显异常形态倾向",
        }

    raw = {
        "protocol": protocol,
        "image_quality": "fair",
        "score": score,
        "pattern_label": labels.get(score, ""),
        "reason": "mock 模式下根据协议和图像路径生成稳定粗评分，用于系统流程调试。",
        "uncertain": False,
    }

    return normalize_vlm_score(raw, protocol=protocol, image_path=image_path)


def call_real_vlm_api(image_path: str, protocol: str) -> Dict[str, Any]:
    """
    真实 VLM API 预留接口。

    后续你接在线 API 或本地 VLM 时，只需要改这里：
    1. 读取 image_path 对应图片；
    2. 调用 VLM；
    3. 获得 JSON 文本；
    4. 用 normalize_vlm_score 标准化输出。
    """
    prompt = build_vlm_score_prompt(protocol)

    raise NotImplementedError(
        "真实 VLM API 尚未接入。当前请使用 mock_vlm_score_image。"
    )


def score_image_with_vlm(
    image_path: str,
    protocol: str,
    use_mock: bool = True,
) -> Dict[str, Any]:
    """
    统一入口。
    """
    image_path_obj = Path(image_path)

    if not image_path_obj.exists():
        return {
            "protocol": protocol,
            "image_path": image_path,
            "image_quality": "poor",
            "score": 0,
            "pattern_label": "图像文件不存在",
            "reason": f"未找到图像文件：{image_path}",
            "uncertain": True,
        }

    if use_mock:
        return mock_vlm_score_image(image_path=image_path, protocol=protocol)

    return call_real_vlm_api(image_path=image_path, protocol=protocol)
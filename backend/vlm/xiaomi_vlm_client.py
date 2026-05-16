# backend/vlm/xiaomi_vlm_client.py

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


def _get_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在：{image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        suffix = path.suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif suffix == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/png"

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_b64}"


def _strip_code_fence(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    return text


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = _strip_code_fence(text)

    if not text:
        return {
            "visual_support": "uncertain",
            "finding": "VLM 未返回有效文本内容",
            "evidence": "接口返回内容为空，暂不能形成图像侧判断",
            "confidence": 0.0,
            "hallucination_flags": ["empty_vlm_response"],
        }

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {
        "visual_support": "uncertain",
        "finding": "VLM 返回内容不是合法 JSON",
        "evidence": text[:500],
        "confidence": 0.0,
        "hallucination_flags": ["invalid_json_response"],
    }


def _normalize_vlm_result(
    result: Dict[str, Any],
    region_info: Dict[str, Any],
    raw_text: str,
) -> Dict[str, Any]:
    visual_support = str(result.get("visual_support", "uncertain")).strip().lower()

    if visual_support not in {"support", "not_support", "uncertain"}:
        visual_support = "uncertain"

    try:
        confidence = float(result.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))

    hallucination_flags = result.get("hallucination_flags", [])
    if not isinstance(hallucination_flags, list):
        hallucination_flags = [str(hallucination_flags)]

    return {
    "patient_id": region_info.get("patient_id"),
    "region_id": region_info.get("region_id"),
    "region_name": region_info.get("region_name"),
    "protocol": region_info.get("protocol"),
    "crop_path": region_info.get("crop_path"),
    "source_image_path": region_info.get("source_image_path"),
    "visual_morphology": str(result.get("visual_morphology", "")),
    "visual_support": visual_support,
    "finding": str(result.get("finding", "图像侧未形成明确判断")),
    "evidence": str(result.get("evidence", "模型未返回明确图像证据")),
    "confidence": confidence,
    "hallucination_flags": hallucination_flags,
    "raw_vlm_response": raw_text,
}


def call_xiaomi_vlm_region_answer(
    region_info: Dict[str, Any],
    prompt: str,
) -> Dict[str, Any]:
    """
    调用小米 MiMo 多模态模型，对单个 ARM 协议阶段图像生成结构化图像侧辅助解释。

    注意：
    - 不输出临床诊断；
    - 不给治疗建议；
    - 不修改 cluster；
    - 只判断当前图像是否支持当前协议阶段的功能状态。
    """

    api_key = (
        _get_env("XIAOMI_API_KEY")
        or _get_env("MIMO_API_KEY")
    )

    base_url = (
        _get_env("XIAOMI_BASE_URL")
        or _get_env("MIMO_BASE_URL")
        or "https://api.xiaomimimo.com/v1"
    )

    model = (
        _get_env("XIAOMI_VLM_MODEL")
        or _get_env("MIMO_VLM_MODEL")
        or _get_env("XIAOMI_MODEL")
        or _get_env("MIMO_MODEL")
        or "mimo-v2-omni"
    )

    if not api_key:
        raise RuntimeError("未配置 XIAOMI_API_KEY 或 MIMO_API_KEY。")

    crop_path = region_info.get("crop_path")
    if not crop_path:
        raise RuntimeError("region_info 中缺少 crop_path，无法调用 VLM。")

    image_data_url = _image_to_data_url(str(crop_path))

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    strict_prompt = f"""
{prompt}

你必须只输出一个 JSON 对象。
禁止输出 Markdown。
禁止输出代码块。
禁止输出解释性文字。
禁止在 JSON 前后添加任何内容。

请先描述图像形态，再判断是否支持当前协议阶段的功能状态。

JSON 格式必须完全如下：

{{
  "visual_morphology": "描述当前热力图中可见的颜色分布、压力带形态、增强或减弱趋势，不写诊断",
  "visual_support": "support",
  "finding": "一句话描述图像侧辅助发现",
  "evidence": "只描述图像中可见的压力热图形态证据，不写诊断",
  "confidence": 0.0,
  "hallucination_flags": []
}}

字段要求：
- visual_morphology 必须描述图像形态。
- visual_support 只能是 support、not_support、uncertain 三者之一。
- finding 必须是字符串。
- evidence 必须是字符串。
- confidence 必须是 0 到 1 之间的小数。
- hallucination_flags 必须是数组。

判断规则：
1. 只能判断当前输入图像。
2. 不得引用其他患者、其他协议阶段或训练数据。
3. 不得输出疾病诊断。
4. 不得给治疗建议。
5. 如果无法判断是否支持功能异常，visual_support 可以为 uncertain。
6. 即使 visual_support 为 uncertain，也必须给出 visual_morphology，描述图像中实际可见的形态。
7. 不要因为缺少临床数值就拒绝描述图像形态。
"""

    completion = client.chat.completions.create(
    model=model,
    temperature=0,
    max_tokens=800,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": strict_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                },
            ],
        }
    ],
)

    message = completion.choices[0].message

    raw_text = message.content or ""

    if not raw_text:
        try:
            raw_text = str(message.model_dump())
        except Exception:
            raw_text = str(message)

    result = _extract_json_object(raw_text)

    return _normalize_vlm_result(
        result=result,
        region_info=region_info,
        raw_text=raw_text,
    )
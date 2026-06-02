# backend/report/llm_client.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import os
import json
import re

from dotenv import load_dotenv

from backend.report.llm_report import (
    generate_rule_based_report,
    safe_dict,
    safe_list,
)


# ============================================================
# 显式读取项目根目录 .env
# backend/report/llm_client.py -> parents[2] = 项目根目录
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=True)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
    }


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "rule").strip().lower()


def _get_xiaomi_api_key() -> str | None:
    return (
        os.getenv("MIMO_API_KEY")
        or os.getenv("XIAOMI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def _get_xiaomi_base_url() -> str:
    return (
        os.getenv("MIMO_BASE_URL")
        or os.getenv("XIAOMI_BASE_URL")
        or "https://api.xiaomimimo.com/v1"
    )


def _get_xiaomi_model() -> str:
    return (
        os.getenv("MIMO_MODEL")
        or os.getenv("XIAOMI_MODEL")
        or os.getenv("XIAOMI_VLM_MODEL")
        or "mimo-v2.5"
    )


def _should_use_real_api() -> bool:
    """
    真实 API 开关判断。

    推荐 .env 显式写：
        LLM_ENABLE_REAL_API=1

    兼容策略：
    - 如果 LLM_ENABLE_REAL_API 明确写了 0，则强制不用真实 API。
    - 如果 LLM_ENABLE_REAL_API 明确写了 1，则使用真实 API。
    - 如果没写 LLM_ENABLE_REAL_API，但 provider=xiaomi 且有 key，也允许走真实 API。
      这样可以避免你漏写开关时一直回退规则版。
    """
    raw_enable = os.getenv("LLM_ENABLE_REAL_API")
    provider = _get_provider()
    has_key = bool(_get_xiaomi_api_key())

    if raw_enable is not None:
        return _env_bool("LLM_ENABLE_REAL_API", default=False)

    return provider in {"xiaomi", "mimo", "xiaomi_mimo"} and has_key


def get_llm_runtime_status() -> Dict[str, Any]:
    """
    可选：给前端调试显示用。
    不返回完整 API key，避免泄漏。
    """
    key = _get_xiaomi_api_key()

    if key:
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 12 else "***"
    else:
        masked_key = "未读取到"

    return {
        "env_path": str(ROOT_DIR / ".env"),
        "env_exists": (ROOT_DIR / ".env").exists(),
        "enable_raw": os.getenv("LLM_ENABLE_REAL_API"),
        "use_real_api": _should_use_real_api(),
        "provider": _get_provider(),
        "base_url": _get_xiaomi_base_url(),
        "model": _get_xiaomi_model(),
        "api_key": masked_key,
    }


def compact_metric_judgements(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    只保留 LLM 解释需要的字段，避免把 raw_row 全量传给模型。
    """
    output = []

    for item in safe_list(items):
        item = safe_dict(item)
        output.append(
            {
                "metric": item.get("metric"),
                "value": item.get("value"),
                "status": item.get("status"),
                "state_text": item.get("state_text"),
                "low": item.get("low"),
                "high": item.get("high"),
                "center": item.get("center"),
                "sex": item.get("sex"),
                "reference_group": item.get("reference_group"),
            }
        )

    return output


def compact_rag_chunks(
    chunks: List[Dict[str, Any]],
    max_chunks: int = 4,
) -> List[Dict[str, Any]]:
    """
    控制 RAG 证据长度，避免 prompt 过长。
    """
    output = []

    for chunk in safe_list(chunks)[:max_chunks]:
        chunk = safe_dict(chunk)

        text = chunk.get("chunk_text") or chunk.get("text") or ""
        text = str(text)

        if len(text) > 800:
            text = text[:800] + "..."

        output.append(
            {
                "title": chunk.get("title"),
                "source": chunk.get("source"),
                "matched_terms": chunk.get("matched_terms"),
                "matched_tags": chunk.get("matched_tags"),
                "score": chunk.get("score"),
                "chunk_text": text,
            }
        )

    return output


def sanitize_context_for_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    给真实 LLM 的最小化输入：
    - 不传患者姓名；
    - 不传图像路径；
    - 不传本地文件路径；
    - 不传 raw_row 全量；
    - 指标正常/异常只使用 feature_state_extractor.py 的结果。
    """
    context = safe_dict(context)
    rag = safe_dict(context.get("rag"))

    return {
        "patient_id": str(context.get("patient_id", "")),
        "ai_result": safe_dict(context.get("ai_result")),
        "phenotype": safe_dict(context.get("phenotype")),
        "abnormal_metrics": compact_metric_judgements(
            safe_list(context.get("abnormal_metrics"))
        ),
        "feature_states": safe_list(context.get("feature_states")),
        "rair_features": safe_dict(context.get("rair_features")),
        "rome_iv": safe_dict(context.get("rome_iv")),
        "rag": {
            "input_features": safe_dict(rag.get("input_features")),
            "retrieved_chunks": compact_rag_chunks(
                safe_list(rag.get("retrieved_chunks"))
            ),
            "explanation": safe_dict(rag.get("explanation")),
        },
        "kg_paths": safe_list(context.get("kg_paths"))[:8],
    }


def build_report_prompt(context: Dict[str, Any]) -> str:
    sanitized = sanitize_context_for_llm(context)

    prompt = f"""
你是一个用于科研辅助解释的 ARM（肛门直肠测压）功能表型分析助手。

请根据给定结构化 JSON 生成一份中文科研解释报告。

重要边界：
1. 你不能改变患者的 AI 分型结果。
2. 你不能重新判断医院指标正常/异常，必须完全使用 JSON 中 abnormal_metrics 和 feature_states。
3. 你不能输出临床诊断、治疗建议、用药建议或确定性医学结论。
4. 你必须明确说明：该报告仅用于科研辅助分析，不用于临床诊断或治疗决策。
5. 如果群体表型名称和个体异常指标不完全一致，请说明“该患者可能存在个体层面的偏离群体主画像特征”，不要强行解释成完全一致。
6. 如果 RAG 证据不足、KG 路径缺失、RAIR 缺失或指标缺失，需要写入不确定性说明。
7. 报告要稳健、克制、像毕业论文系统展示中的科研解释，不要夸大模型能力。
8. 不要输出患者姓名、文件路径、API 信息或任何隐私字段。
9. 输出时不要直接出现 JSON 字段名，例如 abnormal_metrics、feature_states、rair_features、rome_iv、kg_paths；请转换成自然中文表达。
请按以下结构输出 Markdown：

### 分析摘要
用 1 段话说明 AI Cluster、表型名称、置信度和稳定性。

### 关键发现
用项目符号列出 3-8 条异常指标，必须包含：
- 指标名
- 患者数值
- 参考范围来源
- 异常方向
- 简短科研解释

### 可能机制
结合 feature_states、RAIR、Rome IV proxy、KG 路径解释可能机制。
注意使用“提示”“可能”“倾向于”这些表述。

### 文献支持
结合 RAG 召回证据总结，不要编造不存在的文献。
如果证据不足，明确说明。

### 科研建议
只给科研分析建议，例如复核、与群体画像对比、结合 RAIR/Rome IV proxy，不给临床治疗建议。

### 不确定性说明
列出模型分型、指标缺失、边界患者、RAG/KG 证据不足等限制。

结构化 JSON 如下：
注意：JSON 仅供你内部理解。最终报告中不要直接暴露 JSON 字段名、变量名或程序字段。
{json.dumps(sanitized, ensure_ascii=False, indent=2)}
"""

    return prompt.strip()


def strip_code_fence(text: str) -> str:
    """
    避免模型把整份报告包在 ```markdown 里。
    """
    text = str(text).strip()

    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def generate_xiaomi_mimo_report(context: Dict[str, Any]) -> str:
    """
    使用 Xiaomi MiMo OpenAI 兼容接口生成报告。
    """
    from openai import OpenAI

    api_key = _get_xiaomi_api_key()
    base_url = _get_xiaomi_base_url()
    model = _get_xiaomi_model()

    if not api_key:
        raise RuntimeError(
            "未读取到小米 MiMo API Key。请在 .env 中配置 XIAOMI_API_KEY 或 MIMO_API_KEY。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    prompt = build_report_prompt(context)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 MiMo，也是一个严谨的中文科研辅助解释助手。"
                    "你只能基于输入 JSON 做科研解释。"
                    "你不能改变患者 AI 分型，不能重新判断医院参考范围，"
                    "不能给出临床诊断或治疗建议。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
        top_p=0.9,
        max_tokens=3000,
        stream=False,
    )

    try:
        text = completion.choices[0].message.content or ""
    except Exception:
        text = str(completion)

    return strip_code_fence(text)


def generate_llm_report(context: Dict[str, Any]) -> str:
    """
    统一入口：
    - LLM_ENABLE_REAL_API=1 且 LLM_PROVIDER=xiaomi 时调用小米 MiMo；
    - 如果 LLM_ENABLE_REAL_API 缺失，但 provider=xiaomi 且 key 存在，也允许调用；
    - 否则使用规则版报告；
    - 真实 API 失败时自动回退规则版，保证页面不断。
    """
    provider = _get_provider()
    use_real_api = _should_use_real_api()

    if not use_real_api:
        return generate_rule_based_report(context)

    try:
        if provider in {"xiaomi", "mimo", "xiaomi_mimo"}:
            return generate_xiaomi_mimo_report(context)

        return generate_rule_based_report(context)

    except Exception as e:
        fallback = generate_rule_based_report(context)
        status = get_llm_runtime_status()

        return (
            "### LLM API 调用失败，已回退为规则版报告\n\n"
            f"错误信息：`{str(e)}`\n\n"
            "**当前 LLM 配置：**\n\n"
            f"- env_path: `{status.get('env_path')}`\n"
            f"- env_exists: `{status.get('env_exists')}`\n"
            f"- enable_raw: `{status.get('enable_raw')}`\n"
            f"- use_real_api: `{status.get('use_real_api')}`\n"
            f"- provider: `{status.get('provider')}`\n"
            f"- base_url: `{status.get('base_url')}`\n"
            f"- model: `{status.get('model')}`\n"
            f"- api_key: `{status.get('api_key')}`\n\n"
            "---\n\n"
            f"{fallback}"
        )
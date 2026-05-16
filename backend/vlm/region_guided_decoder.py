# backend/vlm/region_guided_decoder.py
import os
from typing import Dict, Any, List
from .functional_regions import crop_all_regions
from .xiaomi_vlm_client import call_xiaomi_vlm_region_answer

def build_region_prompt(region_info: Dict[str, Any]) -> str:
    return f"""
你是 ARM 功能测压图像的科研辅助解释模型。

请只观察当前输入图像对应的【{region_info["region_name"]}】。
不要根据其他协议阶段、疾病常识或统计先验进行推断。

任务：判断该图像是否支持以下功能状态：
{region_info["target_feature"]}

问题：
{region_info["question"]}

只能从以下三类中选择：
1. support：图像区域存在可见支持证据；
2. not_support：图像区域未见支持证据；
3. uncertain：图像质量不足、区域不清晰或无法判断。

请只输出 JSON。
不要输出临床诊断。
不要给治疗建议。
"""


def mock_vlm_region_answer(region_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    当前先用 mock 跑通流程。
    后续替换为真实 VLM API 调用。
    """

    region_id = region_info.get("region_id")

    if region_id == "resting_phase":
        visual_support = "uncertain"
        finding = "静息阶段图像侧证据暂不明确"
        evidence = "当前 mock 阶段未进行真实视觉模型判断，仅用于验证工程流程"
        confidence = 0.50

    elif region_id == "squeeze_phase":
        visual_support = "uncertain"
        finding = "缩榨阶段图像侧证据暂不明确"
        evidence = "当前 mock 阶段未进行真实视觉模型判断，仅用于验证工程流程"
        confidence = 0.50

    elif region_id == "defecation_phase":
        visual_support = "support"
        finding = "排便模拟阶段可能存在推进压力响应偏弱"
        evidence = "mock 结果：目标协议图像提示压力增强趋势可能不足"
        confidence = 0.72

    elif region_id == "rair_phase":
        visual_support = "uncertain"
        finding = "RAIR阶段图像证据不足"
        evidence = "当前 mock 阶段无法可靠判断松弛与恢复趋势"
        confidence = 0.55

    elif region_id == "cough_phase":
        visual_support = "uncertain"
        finding = "咳嗽反射阶段图像证据暂不明确"
        evidence = "当前 mock 阶段未进行真实视觉模型判断，仅用于验证工程流程"
        confidence = 0.50

    else:
        visual_support = "uncertain"
        finding = "未知协议阶段，暂不形成图像侧判断"
        evidence = "路径中未能识别 RestPressure / Contraction / Defecation / rair / Cough"
        confidence = 0.40

    return {
        "patient_id": region_info["patient_id"],
        "region_id": region_info["region_id"],
        "region_name": region_info["region_name"],
        "protocol": region_info.get("protocol"),
        "crop_path": region_info["crop_path"],
        "source_image_path": region_info.get("source_image_path"),
        "visual_support": visual_support,
        "finding": finding,
        "evidence": evidence,
        "confidence": confidence,
        "hallucination_flags": [],
        "prompt": build_region_prompt(region_info),
    }


def generate_region_findings(
    patient_id: str,
    image_path: str,
    output_dir: str = "outputs/vlm_region_crops",
) -> List[Dict[str, Any]]:
    region_infos = crop_all_regions(
        image_path=image_path,
        output_dir=output_dir,
        patient_id=patient_id,
    )

    use_real_vlm = str(os.getenv("VLM_ENABLE_REAL_API", "false")).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "是",
    }

    findings = []

    for region_info in region_infos:
        if use_real_vlm:
            prompt = build_region_prompt(region_info)
            finding = call_xiaomi_vlm_region_answer(
                region_info=region_info,
                prompt=prompt,
            )
            finding["prompt"] = prompt
            finding["vlm_provider"] = "xiaomi"
            finding["vlm_mode"] = "real_api"
        else:
            finding = mock_vlm_region_answer(region_info)
            finding["vlm_provider"] = "mock"
            finding["vlm_mode"] = "mock"

        findings.append(finding)

    return findings
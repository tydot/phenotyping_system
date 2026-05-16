# backend/vlm/functional_regions.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from PIL import Image


@dataclass
class FunctionalRegionSpec:
    """
    ARM 单协议图像功能区域定义。

    当前版本不再把一张图横向切成四个阶段。
    因为 preprocessed_features 中每张图本身已经对应一个协议阶段：
    RestPressure / Contraction / Defecation / rair / Cough。
    """

    region_id: str
    region_name: str
    target_feature: str
    question: str


PROTOCOL_REGION_MAP = {
    "restpressure": FunctionalRegionSpec(
        region_id="resting_phase",
        region_name="静息阶段",
        target_feature="静息压异常或肛管高压区改变",
        question="该图像是否支持静息压升高、降低或肛管高压区改变的图像证据？",
    ),
    "contraction": FunctionalRegionSpec(
        region_id="squeeze_phase",
        region_name="缩榨阶段",
        target_feature="最大缩榨压或主动收缩能力异常",
        question="该图像是否支持主动收缩能力不足或缩榨压力异常？",
    ),
    "defecation": FunctionalRegionSpec(
        region_id="defecation_phase",
        region_name="排便模拟阶段",
        target_feature="直肠推进压力不足或排便协调异常",
        question="该图像是否支持直肠推进压力不足或排便协调异常？",
    ),
    "rair": FunctionalRegionSpec(
        region_id="rair_phase",
        region_name="RAIR反射阶段",
        target_feature="RAIR松弛反应或恢复过程异常",
        question="该图像是否存在可见松弛和恢复反应？",
    ),
    "cough": FunctionalRegionSpec(
        region_id="cough_phase",
        region_name="咳嗽反射阶段",
        target_feature="咳嗽诱发压力反应异常",
        question="该图像是否支持咳嗽诱发压力反应不足或异常？",
    ),
}


def infer_protocol_from_path(image_path: str) -> str:
    """
    从路径中推断协议阶段。
    """
    text = str(image_path).lower()

    if "restpressure" in text or "rest" in text or "静息" in text:
        return "restpressure"
    if "contraction" in text or "squeeze" in text or "提肛" in text or "缩榨" in text or "压肛" in text:
        return "contraction"
    if "defecation" in text or "排便" in text:
        return "defecation"
    if "rair" in text:
        return "rair"
    if "cough" in text or "咳嗽" in text:
        return "cough"

    return "unknown"


def build_region_spec_from_image(image_path: str) -> FunctionalRegionSpec:
    protocol = infer_protocol_from_path(image_path)

    if protocol in PROTOCOL_REGION_MAP:
        return PROTOCOL_REGION_MAP[protocol]

    return FunctionalRegionSpec(
        region_id="unknown_phase",
        region_name="未知协议阶段",
        target_feature="未知图像侧功能特征",
        question="该图像对应的 ARM 协议阶段无法从路径中识别，请人工核对。",
    )


def crop_all_regions(
    image_path: str,
    output_dir: str,
    patient_id: str,
):
    """
    对单张协议图像生成一个“整图区域”。

    注意：
    这里保留 crop_all_regions 这个函数名，是为了兼容 region_guided_decoder.py。
    但实际不再横向四等分，而是整张图作为当前协议阶段证据。
    """
    image_path_obj = Path(image_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    if not image_path_obj.exists():
        raise FileNotFoundError(f"图像文件不存在：{image_path}")

    spec = build_region_spec_from_image(str(image_path_obj))

    img = Image.open(image_path_obj).convert("RGB")

    safe_patient_id = str(patient_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    out_path = output_dir_obj / f"{safe_patient_id}_{spec.region_id}.png"
    img.save(out_path)

    return [
        {
            "patient_id": str(patient_id),
            "region_id": spec.region_id,
            "region_name": spec.region_name,
            "target_feature": spec.target_feature,
            "question": spec.question,
            "crop_path": str(out_path),
            "crop_box": None,
            "source_image_path": str(image_path_obj),
            "protocol": infer_protocol_from_path(str(image_path_obj)),
        }
    ]
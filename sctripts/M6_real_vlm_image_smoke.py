from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
import base64
import io
import numpy as np
from PIL import Image
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.xiaomi", override=True, encoding="utf-8-sig")

MANIFEST = ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"

def npy_to_data_url(npy_path):
    arr = np.load(npy_path)

    # 你的 npy 是 (3, 224, 224)，转成 (224, 224, 3)
    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 2, 0))

    arr = np.asarray(arr)

    # 兼容 float32。若在 0~1，转 0~255；若本来接近 0~255，直接截断
    if arr.dtype != np.uint8:
        if np.nanmax(arr) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        mode = "L"
    else:
        mode = "RGB"

    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


def main():
    api_key = os.getenv("XIAOMI_API_KEY", "").strip()
    base_url = os.getenv("XIAOMI_BASE_URL", "").strip()
    model = os.getenv("XIAOMI_VLM_MODEL", "").strip()

    print("api_key:", "OK" if api_key else "MISSING", "len=", len(api_key))
    print("base_url:", base_url)
    print("model:", model)

    df = pd.read_csv(MANIFEST)
    row = df.iloc[0]

    image_path = Path(str(row["feature_path_resolved"]))
    protocol = str(row["protocol"])
    patient_id = row["patient_id"]

    print("\n[测试样本]")
    print("patient_id:", patient_id)
    print("protocol:", protocol)
    print("image_path:", image_path)
    print("exists:", image_path.exists())

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    data_url = npy_to_data_url(image_path)

    prompt = f"""
你是一名辅助分析肛肠测压图像的医学图像质控助手。

当前图像对应检查协议：{protocol}

请只输出 JSON，不要输出解释，不要输出 Markdown。
JSON 格式必须是：
{{
  "vlm_score_raw": 1到4之间的整数,
  "vlm_image_quality": "poor/fair/good/excellent之一",
  "vlm_pattern_label": "一句中文标签",
  "vlm_reason": "一句中文理由",
  "vlm_uncertain": true或false
}}

评分含义：
1 = 图像质量差或模式不清楚
2 = 可用但特征较弱
3 = 图像质量较好且有一定模式特征
4 = 图像质量好且协议相关模式较明显
""".strip()

    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0,
        max_tokens=2048,
    )

    msg = resp.choices[0].message

    print("\n[content]")
    print(msg.content)

    print("\n[finish_reason]")
    print(resp.choices[0].finish_reason)

    if hasattr(msg, "reasoning_content"):
        print("\n[reasoning_content 存在，但后续 pipeline 不使用]")
        rc = getattr(msg, "reasoning_content", None)
        print(str(rc)[:300] if rc else None)


if __name__ == "__main__":
    main()


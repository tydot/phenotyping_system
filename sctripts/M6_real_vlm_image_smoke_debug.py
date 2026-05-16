from pathlib import Path
import os
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env.xiaomi"

load_dotenv(ENV_PATH, override=True, encoding="utf-8-sig")

API_KEY = os.getenv("XIAOMI_API_KEY", "").strip()
BASE_URL = os.getenv("XIAOMI_BASE_URL", "").strip()
MODEL = os.getenv("XIAOMI_VLM_MODEL", "").strip()

MANIFEST = ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"
OUT_DIR = ROOT / "outputs" / "vlm_debug_sent"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def npy_to_rgb_image(npy_path: Path) -> Image.Image:
    arr = np.load(npy_path)

    print("[npy]")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min/max/mean:", float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr)))

    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)

    if not finite.any():
        raise RuntimeError("npy 全是 NaN/Inf，不能转图")

    lo, hi = np.percentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))

    arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def save_and_encode_jpeg(img: Image.Image, out_path: Path) -> str:
    img.save(out_path, format="JPEG", quality=95)
    b = out_path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return "data:image/jpeg;base64," + b64


def call_vlm(data_url: str, protocol: str, style: str):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
你正在查看一张肛肠测压数据转换得到的伪彩色压力热图。

当前协议：{protocol}

请先判断你是否真的看到了图像内容。注意：
- 这不是自然照片；
- 这是彩色热图；
- 如果看到彩色块、红黄绿蓝区域、条带或压力分布，不要说它是黑图；
- 只有在整张图完全黑色或空白时，才允许说黑图。

只输出 JSON，不要 Markdown。

格式：
{{
  "image_seen": true,
  "is_black_or_blank": false,
  "visible_description": "一句话描述你看到的颜色和结构",
  "vlm_score_raw": 1,
  "vlm_image_quality": "poor",
  "vlm_pattern_label": "一句中文标签",
  "vlm_reason": "一句中文理由",
  "vlm_uncertain": true
}}

vlm_score_raw 取 1 到 4：
1 = 图像质量差、模式不可判断
2 = 图像质量一般、能看到部分模式
3 = 图像质量较好、模式较清楚
4 = 图像质量很好、模式清晰连续
""".strip()

    if style == "object":
        image_part = {
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        }
    elif style == "string":
        image_part = {
            "type": "image_url",
            "image_url": data_url
        }
    else:
        raise ValueError(style)

    print(f"\n[API call style={style}]")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    image_part
                ]
            }
        ],
        temperature=0,
        max_tokens=800,
    )

    msg = resp.choices[0].message
    print("[finish_reason]", resp.choices[0].finish_reason)
    print("[content]")
    print(msg.content)

    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        print("[reasoning_content 前 300 字，仅调试]")
        print(reasoning[:300])


def main():
    print("ROOT:", ROOT)
    print("ENV_PATH:", ENV_PATH, "exists:", ENV_PATH.exists())
    print("api_key:", "OK" if API_KEY else "MISSING", "len=", len(API_KEY))
    print("base_url:", BASE_URL)
    print("model:", MODEL)
    print("manifest:", MANIFEST, "exists:", MANIFEST.exists())

    if not API_KEY or not BASE_URL or not MODEL:
        raise RuntimeError("请先检查 .env.xiaomi")

    df = pd.read_csv(MANIFEST)
    row = df.iloc[0]

    protocol = str(row["protocol"])
    image_path = Path(str(row["feature_path_resolved"]))

    print("\n[测试样本]")
    print("patient_id:", row["patient_id"])
    print("protocol:", protocol)
    print("image_path:", image_path)
    print("exists:", image_path.exists())

    img = npy_to_rgb_image(image_path)

    out_img = OUT_DIR / "sent_000_pid2_Contraction_rank0.jpg"
    data_url = save_and_encode_jpeg(img, out_img)

    print("\n[已保存实际发送图片]")
    print(out_img)
    print("exists:", out_img.exists())
    print("file_size:", out_img.stat().st_size)
    print("data_url_head:", data_url[:50])

    call_vlm(data_url, protocol, style="object")

    try:
        call_vlm(data_url, protocol, style="string")
    except Exception as e:
        print("\n[string style failed]")
        print(type(e).__name__, str(e))


if __name__ == "__main__":
    main()

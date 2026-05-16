import argparse
import base64
import json
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")
IN_MANIFEST = PROJECT_ROOT / "outputs" / "vlm_real_smoke" / "real_vlm_smoke_manifest.csv"
OUT_CSV = PROJECT_ROOT / "outputs" / "vlm_real_smoke" / "real_vlm_image_scores_smoke.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"


PROTOCOL_RUBRIC = {
    "RestPressure": "判断静息压力图像是否清晰、是否包含稳定静息压力形态，不做疾病诊断。",
    "Contraction": "判断提肛收缩图像是否清晰、是否包含收缩增强或收缩不足等协议相关形态，不做疾病诊断。",
    "Defecation": "判断模拟排便图像是否清晰、是否包含排便推进、矛盾收缩或协调性相关形态，不做疾病诊断。",
    "Cough": "判断咳嗽反射图像是否清晰、是否包含咳嗽诱发压力变化，不做疾病诊断。",
    "rair": "判断 RAIR 图像是否清晰、是否包含球囊扩张后的反射性变化线索，不做疾病诊断。",
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(protocol):
    rubric = PROTOCOL_RUBRIC.get(protocol, "判断该图像是否清晰、是否包含当前协议相关的压力变化形态，不做疾病诊断。")

    return f"""
你是一个肛肠测压图像质控与粗粒度模式识别助手。你只能根据图像本身判断该帧是否适合用于后续表征学习加权，不要给出疾病诊断。

当前协议: {protocol}
任务说明: {rubric}

请严格输出 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。

评分定义:
0 = 图像不可读、几乎无法判断、或与当前协议无关
1 = 图像质量一般或协议相关证据很弱
2 = 图像可读，存在一定协议相关形态
3 = 图像清晰，协议相关形态明显，适合作为代表帧

JSON 字段:
{{
  "vlm_score_raw": 0到3的整数,
  "vlm_image_quality": "poor/fair/good",
  "vlm_pattern_label": "不超过20字的中文标签",
  "vlm_reason": "不超过40字的中文理由",
  "vlm_uncertain": true或false
}}
""".strip()


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON found: {text[:200]}")

    return json.loads(m.group(0))


def call_ollama(model, png_path, protocol):
    payload = {
        "model": model,
        "prompt": build_prompt(protocol),
        "images": [encode_image(png_path)],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()

    data = resp.json()
    raw_text = data.get("response", "")
    parsed = extract_json(raw_text)

    score = int(parsed.get("vlm_score_raw", 0))
    score = max(0, min(3, score))

    quality = str(parsed.get("vlm_image_quality", "fair")).strip().lower()
    if quality not in ["poor", "fair", "good"]:
        quality = "fair"

    label = str(parsed.get("vlm_pattern_label", "未明确")).strip()
    reason = str(parsed.get("vlm_reason", "")).strip()
    uncertain = bool(parsed.get("vlm_uncertain", False))

    return {
        "vlm_score_raw": score,
        "vlm_image_quality": quality,
        "vlm_pattern_label": label,
        "vlm_reason": reason,
        "vlm_uncertain": uncertain,
        "vlm_mode": f"real_ollama_{model}",
        "vlm_raw_response": raw_text,
        "vlm_error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama vision model name, e.g. llava")
    args = parser.parse_args()

    df = pd.read_csv(IN_MANIFEST)
    df = df[df["png_exists"] == True].copy().reset_index(drop=True)

    print("[1] 输入 smoke manifest:", df.shape)
    print("protocol counts:")
    print(df["protocol"].value_counts())

    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        row = r.to_dict()

        protocol = str(r["protocol"])
        png_path = str(r["vlm_png_path"])

        try:
            result = call_ollama(args.model, png_path, protocol)
        except Exception as e:
            result = {
                "vlm_score_raw": 0,
                "vlm_image_quality": "poor",
                "vlm_pattern_label": "模型调用失败",
                "vlm_reason": repr(e)[:120],
                "vlm_uncertain": True,
                "vlm_mode": f"real_ollama_{args.model}",
                "vlm_raw_response": "",
                "vlm_error": repr(e),
            }

        score = int(result["vlm_score_raw"])
        score_norm = score / 3.0
        reweight_factor = 0.25 + 0.75 * score_norm

        row["image_path"] = row.get("vlm_input_path", row.get("feature_path_resolved", row.get("source_npy_path")))
        row["vlm_score_raw"] = score
        row["vlm_score_norm"] = score_norm
        row["vlm_reweight_factor"] = reweight_factor
        row.update(result)

        rows.append(row)

    out = pd.DataFrame(rows)

    keep_first = [
        "patient_id",
        "protocol",
        "rank",
        "image_path",
        "attention_weight",
        "centroid_score",
        "topk",
        "temperature",
        "vlm_score_raw",
        "vlm_image_quality",
        "vlm_pattern_label",
        "vlm_reason",
        "vlm_uncertain",
        "vlm_mode",
        "vlm_score_norm",
        "vlm_reweight_factor",
        "vlm_png_path",
        "vlm_error",
        "vlm_raw_response",
    ]

    cols = [c for c in keep_first if c in out.columns] + [c for c in out.columns if c not in keep_first]
    out = out[cols]

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n[DONE]")
    print("输出:", OUT_CSV)
    print("\nscore 分布:")
    print(out.groupby(["protocol", "vlm_score_raw"]).size())
    print("\nuncertain 比例:")
    print(out.groupby("protocol")["vlm_uncertain"].mean())
    print("\nerror 数:", int((out["vlm_error"].astype(str) != "").sum()))


if __name__ == "__main__":
    main()
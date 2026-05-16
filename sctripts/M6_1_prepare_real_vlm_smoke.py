import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")
M1_MANIFEST = PROJECT_ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "vlm_real_smoke"
PNG_DIR = OUT_DIR / "pngs"
OUT_MANIFEST = OUT_DIR / "real_vlm_smoke_manifest.csv"
CONTACT_SHEET = OUT_DIR / "preview_contact_sheet.png"

N_PER_PROTOCOL = 20
RANDOM_STATE = 42

PROTOCOL_ORDER = ["RestPressure", "Contraction", "Defecation", "Cough", "rair"]


def npy_to_uint8_image(arr: np.ndarray):
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros((224, 224, 3), dtype=np.uint8), {
            "min": np.nan,
            "max": np.nan,
            "p1": np.nan,
            "p99": np.nan,
            "mode": "all_nonfinite",
        }

    vals = arr[finite]
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    p1 = float(np.percentile(vals, 1))
    p99 = float(np.percentile(vals, 99))

    if mn >= 0 and mx <= 1.5:
        out = np.clip(arr, 0, 1) * 255.0
        mode = "scale_0_1"
    elif mn >= 0 and mx <= 255:
        out = np.clip(arr, 0, 255)
        mode = "scale_0_255"
    else:
        denom = max(p99 - p1, 1e-6)
        out = (arr - p1) / denom
        out = np.clip(out, 0, 1) * 255.0
        mode = "robust_percentile_1_99"

    out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
    out = out.astype(np.uint8)

    return out, {
        "min": mn,
        "max": mx,
        "p1": p1,
        "p99": p99,
        "mode": mode,
    }


def make_contact_sheet(rows, thumb_size=(160, 160), cols=5):
    imgs = []
    labels = []

    for _, r in rows.head(25).iterrows():
        img = Image.open(r["vlm_png_path"]).convert("RGB")
        img.thumbnail(thumb_size)
        canvas = Image.new("RGB", thumb_size, "white")
        x = (thumb_size[0] - img.width) // 2
        y = (thumb_size[1] - img.height) // 2
        canvas.paste(img, (x, y))
        imgs.append(canvas)
        labels.append(f'{r["protocol"]} | rank={r["rank"]}')

    if not imgs:
        return

    label_h = 24
    rows_n = int(np.ceil(len(imgs) / cols))
    sheet = Image.new(
        "RGB",
        (cols * thumb_size[0], rows_n * (thumb_size[1] + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for i, img in enumerate(imgs):
        c = i % cols
        rr = i // cols
        x = c * thumb_size[0]
        y = rr * (thumb_size[1] + label_h)
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + thumb_size[1] + 4), labels[i], fill=(0, 0, 0))

    sheet.save(CONTACT_SHEET)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] 读取 M1 topk4 manifest")
    df = pd.read_csv(M1_MANIFEST)
    print("manifest:", df.shape)
    print("columns:", list(df.columns))

    path_col = "feature_path_resolved" if "feature_path_resolved" in df.columns else "vlm_input_path"
    print("使用路径列:", path_col)

    sampled = []
    for protocol in PROTOCOL_ORDER:
        sub = df[df["protocol"].astype(str) == protocol].copy()
        n = min(N_PER_PROTOCOL, len(sub))
        if n > 0:
            sampled.append(sub.sample(n=n, random_state=RANDOM_STATE))
        print(f"{protocol}: total={len(sub)}, sampled={n}")

    out = pd.concat(sampled, axis=0).reset_index(drop=True)

    print("\n[2] 转换 npy -> png")
    records = []
    for i, r in out.iterrows():
        src = Path(str(r[path_col]))
        patient_id = str(r["patient_id"])
        protocol = str(r["protocol"])
        rank = int(r["rank"])

        png_name = f"{i:04d}_pid{patient_id}_{protocol}_rank{rank}.png"
        png_path = PNG_DIR / png_name

        rec = r.to_dict()
        rec["source_npy_path"] = str(src)
        rec["vlm_png_path"] = str(png_path)

        try:
            arr = np.load(src)
            img_arr, stat = npy_to_uint8_image(arr)
            Image.fromarray(img_arr).save(png_path)
            rec["png_exists"] = True
            rec["npy_shape"] = str(arr.shape)
            rec["npy_dtype"] = str(arr.dtype)
            rec.update({f"npy_{k}": v for k, v in stat.items()})
        except Exception as e:
            rec["png_exists"] = False
            rec["error"] = repr(e)

        records.append(rec)

    out2 = pd.DataFrame(records)

    out2.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")
    make_contact_sheet(out2[out2["png_exists"] == True])

    print("\n[DONE]")
    print("输出 manifest:", OUT_MANIFEST)
    print("预览图:", CONTACT_SHEET)
    print("成功 PNG:", int(out2["png_exists"].sum()), "/", len(out2))
    print("\n各协议数量:")
    print(out2["protocol"].value_counts())


if __name__ == "__main__":
    main()
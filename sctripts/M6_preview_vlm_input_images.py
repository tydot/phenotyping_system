from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"
OUT_DIR = ROOT / "outputs" / "vlm_debug_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def npy_to_uint8_image(path):
    arr = np.load(path)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    # 单通道转 RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)

    # 若已经是 0-1 或 0-255，优先保留；否则用分位数拉伸
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("array has no finite values")

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if vmin >= 0 and vmax <= 1.5:
        arr = arr * 255.0
    elif vmin >= 0 and vmax <= 255:
        arr = arr
    else:
        lo, hi = np.percentile(arr[finite], [1, 99])
        if hi <= lo:
            lo, hi = vmin, vmax
        arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

df = pd.read_csv(MANIFEST)
path_col = "feature_path_resolved"

print("manifest:", MANIFEST)
print("shape:", df.shape)
print("out:", OUT_DIR)

for i, row in df.head(12).iterrows():
    p = Path(str(row[path_col]))
    img = npy_to_uint8_image(p)
    name = f"{i:03d}_pid{row['patient_id']}_{row['protocol']}_rank{row['rank']}.png"
    out = OUT_DIR / name
    img.save(out)
    arr = np.load(p)
    print(i, row["patient_id"], row["protocol"], p.name, arr.shape, arr.dtype, "->", out)

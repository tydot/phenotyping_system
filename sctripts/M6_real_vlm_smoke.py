from pathlib import Path
import pandas as pd

from M6_real_vlm_common import ROOT, call_xiaomi_vlm


MANIFEST = ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"


def main():
    print("[1] 加载 M1 topk4 manifest")
    df = pd.read_csv(MANIFEST)
    print("manifest:", MANIFEST)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    path_col = None
    for c in ["feature_path_resolved", "feature_path_raw", "vlm_input_path", "image_path", "filepath"]:
        if c in df.columns:
            path_col = c
            break

    if path_col is None:
        raise RuntimeError("manifest 中没有找到图像路径列")

    print("使用路径列:", path_col)

    for i in range(min(3, len(df))):
        row = df.iloc[i]
        image_path = Path(str(row[path_col]))
        protocol = str(row["protocol"])

        print("\n==========")
        print("index:", i)
        print("patient_id:", row["patient_id"])
        print("protocol:", protocol)
        print("image_path:", image_path)
        print("exists:", image_path.exists())

        if not image_path.exists():
            print("[SKIP] 文件不存在")
            continue

        result = call_xiaomi_vlm(image_path, protocol)
        print("VLM result:")
        for k, v in result.items():
            if k != "vlm_raw_response":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

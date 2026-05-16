from pathlib import Path
import time
import pandas as pd

from M6_real_vlm_common import call_xiaomi_vlm


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"
OUT_DIR = ROOT / "outputs" / "m6_real_vlm_smoke_multi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "m6_real_vlm_smoke_multi_scores.csv"


def main():
    print("[1] 加载 manifest")
    df = pd.read_csv(MANIFEST)
    print("manifest:", MANIFEST)
    print("shape:", df.shape)
    print("protocol counts:")
    print(df["protocol"].value_counts())

    path_col = "feature_path_resolved"
    if path_col not in df.columns:
        path_col = "vlm_input_path"

    print("使用路径列:", path_col)

    samples = []
    for protocol, g in df.groupby("protocol", sort=False):
        g = g.sort_values(["patient_id", "rank"]).head(2)
        samples.append(g)

    test_df = pd.concat(samples, axis=0).reset_index(drop=True)

    print("\n[2] smoke 样本数:", len(test_df))
    print(test_df[["patient_id", "protocol", "rank", path_col]].to_string(index=False))

    rows = []

    for i, row in test_df.iterrows():
        image_path = Path(str(row[path_col]))
        protocol = str(row["protocol"])

        print("\n==========")
        print("i:", i)
        print("patient_id:", row["patient_id"])
        print("protocol:", protocol)
        print("rank:", row["rank"])
        print("image_path:", image_path)
        print("exists:", image_path.exists())

        try:
            result = call_xiaomi_vlm(image_path, protocol)
            status = "ok"
            error = ""
        except Exception as e:
            result = {
                "vlm_score_raw": None,
                "vlm_image_quality": None,
                "vlm_pattern_label": "",
                "vlm_reason": "",
                "vlm_uncertain": True,
                "vlm_mode": "real_xiaomi",
                "finish_reason": "",
            }
            status = "error"
            error = repr(e)

        print("status:", status)
        print(result)
        if error:
            print("error:", error)

        rows.append({
            "source_index": int(row.name),
            "patient_id": row["patient_id"],
            "protocol": protocol,
            "rank": row["rank"],
            "image_path": str(image_path),
            "status": status,
            "error": error,
            **result,
        })

        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        time.sleep(0.5)

    print("\n[DONE]")
    print("输出:", OUT_CSV)


if __name__ == "__main__":
    main()

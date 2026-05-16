from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")
CSV = PROJECT_ROOT / "outputs" / "vlm_real_smoke" / "real_vlm_image_scores_smoke.csv"


def main():
    df = pd.read_csv(CSV)

    print("CSV:", CSV)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    print("\n[1] vlm_mode 分布")
    print(df["vlm_mode"].value_counts(dropna=False))

    print("\n[2] vlm_score_raw 总体分布")
    print(df["vlm_score_raw"].value_counts(dropna=False).sort_index())

    print("\n[3] 各协议 score 分布")
    print(pd.crosstab(df["protocol"], df["vlm_score_raw"]))

    print("\n[4] 各协议 image_quality 分布")
    print(pd.crosstab(df["protocol"], df["vlm_image_quality"]))

    print("\n[5] 各协议 uncertain 比例")
    print(df.groupby("protocol")["vlm_uncertain"].mean())

    print("\n[6] reweight factor 描述")
    print(df.groupby("protocol")["vlm_reweight_factor"].describe())

    print("\n[7] error 数")
    err = df[df["vlm_error"].fillna("").astype(str) != ""]
    print(len(err))
    if len(err) > 0:
        print(err[["patient_id", "protocol", "rank", "vlm_error"]].head(10).to_string(index=False))

    print("\n[8] 示例输出")
    show_cols = [
        "patient_id",
        "protocol",
        "rank",
        "vlm_score_raw",
        "vlm_image_quality",
        "vlm_pattern_label",
        "vlm_reason",
        "vlm_uncertain",
    ]
    print(df[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
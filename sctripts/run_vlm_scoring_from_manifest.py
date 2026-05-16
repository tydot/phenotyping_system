# sctripts/run_vlm_scoring_from_manifest.py
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import sys
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vlm.vlm_client import score_image_with_vlm


DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "vlm" / "m1_topk4_vlm_manifest.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "vlm" / "vlm_image_scores_m1_topk4_mock.csv"


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--real_vlm", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="调试用；0 表示全部运行")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest：{manifest_path}")

    df = read_csv_safely(manifest_path)

    required = [
        "patient_id",
        "protocol",
        "rank",
        "vlm_input_path",
        "attention_weight",
        "centroid_score",
        "topk",
        "temperature",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest 缺少字段：{missing}，当前字段：{list(df.columns)}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows = []
    use_mock = not args.real_vlm

    for idx, row in df.iterrows():
        patient_id = str(row["patient_id"]).strip()
        protocol = str(row["protocol"]).strip()
        rank = int(row["rank"])
        image_path = str(row["vlm_input_path"]).strip()

        result = score_image_with_vlm(
            image_path=image_path,
            protocol=protocol,
            use_mock=use_mock,
        )

        rows.append(
            {
                "patient_id": patient_id,
                "protocol": protocol,
                "rank": rank,
                "image_path": image_path,
                "attention_weight": row["attention_weight"],
                "centroid_score": row["centroid_score"],
                "topk": row["topk"],
                "temperature": row["temperature"],
                "vlm_score_raw": result.get("score", 0),
                "vlm_image_quality": result.get("image_quality", "unknown"),
                "vlm_pattern_label": result.get("pattern_label", ""),
                "vlm_reason": result.get("reason", ""),
                "vlm_uncertain": result.get("uncertain", True),
                "vlm_mode": "real" if args.real_vlm else "mock",
            }
        )

        if len(rows) % 1000 == 0:
            print(f"[PROGRESS] {len(rows)}/{len(df)}")

    out_df = pd.DataFrame(rows)

    out_df["vlm_score_raw"] = pd.to_numeric(out_df["vlm_score_raw"], errors="coerce").fillna(0)

    # 当前 VLM 评分范围是 0-3。
    # 0 表示不可判断，1-3 表示由弱到强的图像侧形态评分。
    # 为避免 0 分直接把 attention 权重打没，映射到 0.25-1.00。
    out_df["vlm_score_norm"] = out_df["vlm_score_raw"].clip(0, 3) / 3.0
    out_df["vlm_reweight_factor"] = 0.25 + 0.75 * out_df["vlm_score_norm"]

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[OK] VLM score 文件已生成")
    print("输出文件:", out_path)
    print("行数:", len(out_df))
    print("患者数:", out_df["patient_id"].nunique())

    print("\n协议分布:")
    print(out_df["protocol"].value_counts(dropna=False))

    print("\nVLM raw score 分布:")
    print(out_df["vlm_score_raw"].value_counts(dropna=False).sort_index())

    print("\nVLM quality 分布:")
    print(out_df["vlm_image_quality"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
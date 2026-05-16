# backend/vlm/patient_vlm_runner.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Optional
import pandas as pd

from backend.vlm.vlm_client import score_image_with_vlm
from backend.vlm.coarse_label_builder import aggregate_patient_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "vlm"


PROTOCOL_PRIORITY = [
    "RestPressure",
    "Contraction",
    "Defecation",
    "RAIR",
    "Cough",
]


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def select_representative_images(
    image_meta_csv: str,
    patient_ids: Optional[List[str]] = None,
    max_patients: Optional[int] = 50,
) -> pd.DataFrame:
    """
    从 image_meta.csv 中为每个患者每个协议选 1 张代表图。
    当前采用最简单策略：每个 patient_id + protocol 取第一张。
    后续可替换为 top-k 选帧结果中的 rank=0 图像。
    """
    meta_path = Path(image_meta_csv)
    meta = read_csv_safely(meta_path)

    required = ["patient_id", "protocol", "filepath"]
    for col in required:
        if col not in meta.columns:
            raise ValueError(f"image_meta.csv 缺少列：{col}，当前列名为：{list(meta.columns)}")

    meta["patient_id"] = meta["patient_id"].astype(str).str.strip()
    meta["protocol"] = meta["protocol"].astype(str).str.strip()

    if patient_ids:
        patient_ids = [str(x).strip() for x in patient_ids]
        meta = meta[meta["patient_id"].isin(patient_ids)]

    if max_patients:
        keep_ids = meta["patient_id"].drop_duplicates().head(max_patients).tolist()
        meta = meta[meta["patient_id"].isin(keep_ids)]

    rows = []

    for (pid, proto), g in meta.groupby(["patient_id", "protocol"]):
        g = g.sort_values("filepath")
        first = g.iloc[0]

        rows.append(
            {
                "patient_id": pid,
                "protocol": proto,
                "image_path": first["filepath"],
            }
        )

    rep = pd.DataFrame(rows)

    # 协议排序
    rep["protocol_order"] = rep["protocol"].apply(
        lambda x: PROTOCOL_PRIORITY.index(x) if x in PROTOCOL_PRIORITY else 999
    )
    rep = rep.sort_values(["patient_id", "protocol_order"]).drop(columns=["protocol_order"])

    return rep


def run_vlm_scoring(
    image_meta_csv: str,
    out_dir: str = str(DEFAULT_OUT_DIR),
    max_patients: Optional[int] = 50,
    use_mock: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_df = select_representative_images(
        image_meta_csv=image_meta_csv,
        max_patients=max_patients,
    )

    image_score_rows = []

    for _, row in rep_df.iterrows():
        patient_id = row["patient_id"]
        protocol = row["protocol"]
        image_path = row["image_path"]

        result = score_image_with_vlm(
            image_path=image_path,
            protocol=protocol,
            use_mock=use_mock,
        )

        result["patient_id"] = patient_id
        image_score_rows.append(result)

        print(
            f"[VLM] patient={patient_id} protocol={protocol} "
            f"score={result.get('score')} label={result.get('pattern_label')}"
        )

    image_scores_df = pd.DataFrame(image_score_rows)

    image_scores_path = out_dir / "vlm_image_scores.csv"
    image_scores_df.to_csv(image_scores_path, index=False, encoding="utf-8-sig")

    patient_labels_df = aggregate_patient_scores(image_scores_df)

    patient_labels_path = out_dir / "vlm_patient_coarse_labels.csv"
    patient_labels_df.to_csv(patient_labels_path, index=False, encoding="utf-8-sig")

    print("[SAVED]", image_scores_path)
    print("[SAVED]", patient_labels_path)

    return image_scores_df, patient_labels_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_meta_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max_patients", type=int, default=50)
    parser.add_argument("--real_vlm", action="store_true")

    args = parser.parse_args()

    run_vlm_scoring(
        image_meta_csv=args.image_meta_csv,
        out_dir=args.out_dir,
        max_patients=args.max_patients,
        use_mock=not args.real_vlm,
    )
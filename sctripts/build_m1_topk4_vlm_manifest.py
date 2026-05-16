# sctripts/build_m1_topk4_vlm_manifest.py
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")

M1_ATTN_TOPK4 = Path(
    r"H:\windows\图像数据\dataProcess\outputs\sensitivity_attn_centroid_tau007\mapp4\seed0\attn_topk_details.csv"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "vlm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST = OUT_DIR / "m1_topk4_vlm_manifest.csv"


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def resolve_feature_path(raw_path: str) -> str:
    """
    attention 文件里很多路径是 D:\\dataProcess\\...
    但你当前主数据目录在 H:\\windows\\图像数据\\dataProcess
    所以这里做一次路径映射。
    """
    if not isinstance(raw_path, str):
        return ""

    p = raw_path.strip()

    direct = Path(p)
    if direct.exists():
        return str(direct)

    candidates = []

    if p.startswith(r"D:\dataProcess"):
        candidates.append(
            p.replace(
                r"D:\dataProcess",
                r"H:\windows\图像数据\dataProcess",
                1,
            )
        )

    if p.startswith("D:/dataProcess"):
        candidates.append(
            p.replace(
                "D:/dataProcess",
                "H:/windows/图像数据/dataProcess",
                1,
            )
        )

    for c in candidates:
        if Path(c).exists():
            return str(Path(c))

    return p


def main():
    if not M1_ATTN_TOPK4.exists():
        raise FileNotFoundError(f"找不到 M1 top-k4 attention 文件：{M1_ATTN_TOPK4}")

    df = read_csv_safely(M1_ATTN_TOPK4)

    required = ["patient_id", "protocol", "rank", "filepath", "score", "weight", "topk", "temperature"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"attention 文件缺少字段：{missing}，当前字段：{list(df.columns)}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["protocol"] = df["protocol"].astype(str).str.strip()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["centroid_score"] = pd.to_numeric(df["score"], errors="coerce")
    df["attention_weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["topk"] = pd.to_numeric(df["topk"], errors="coerce").astype("Int64")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    df["feature_path_raw"] = df["filepath"].astype(str)
    df["feature_path_resolved"] = df["feature_path_raw"].apply(resolve_feature_path)
    df["feature_exists"] = df["feature_path_resolved"].apply(lambda x: Path(x).exists())

    # 当前 VLM mock 只检查路径是否存在，不真正读图。
    # 后续接真实 VLM 时，这里需要换成 png/jpg 的 image_path。
    df["vlm_input_path"] = df["feature_path_resolved"]

    out_cols = [
        "patient_id",
        "protocol",
        "rank",
        "vlm_input_path",
        "feature_path_raw",
        "feature_path_resolved",
        "feature_exists",
        "centroid_score",
        "attention_weight",
        "topk",
        "temperature",
    ]

    out = df[out_cols].copy()
    out.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")

    n_rows = len(out)
    n_patients = out["patient_id"].nunique()
    n_exists = int(out["feature_exists"].sum())
    exists_rate = n_exists / n_rows if n_rows else 0

    print("[OK] 已生成 M1 top-k4 VLM manifest")
    print("输出文件:", OUT_MANIFEST)
    print("行数:", n_rows)
    print("患者数:", n_patients)
    print("feature_exists:", f"{n_exists}/{n_rows} = {exists_rate:.2%}")
    print("topk 分布:")
    print(out["topk"].value_counts(dropna=False).sort_index())
    print("协议分布:")
    print(out["protocol"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
# sctripts/inspect_cache_alignment.py
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main")
CACHE_DIR = Path(r"H:\windows\图像数据\dataProcess\outputs\cache")

IMAGE_EMB = CACHE_DIR / "image_embeddings.npy"
IMAGE_META = CACHE_DIR / "image_meta.csv"


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"无法读取 CSV: {path}")


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace("\\", "/")


def basename_key(x):
    s = norm_text(x)
    if not s:
        return ""
    return Path(s).name.lower()


def stem_key(x):
    b = basename_key(x)
    if not b:
        return ""
    return re.sub(r"\.[^.]+$", "", b).lower()


def find_manifest_candidates(root: Path):
    cands = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        full = str(p).lower()
        if any(k in name for k in ["manifest", "topk", "m1", "frame"]):
            cands.append(p)
        elif "vlm" in full and "csv" in name:
            cands.append(p)
    return sorted(set(cands), key=lambda x: x.stat().st_mtime, reverse=True)


def guess_path_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["path", "file", "filename", "image", "npy", "png", "jpg"]):
            cols.append(c)
    return cols


def guess_patient_col(df: pd.DataFrame):
    for c in df.columns:
        cl = str(c).lower()
        if cl in ["patient_id", "pid", "patient", "患者id", "患者编号", "id"]:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if "patient" in cl or "pid" in cl:
            return c
    return None


def guess_protocol_col(df: pd.DataFrame):
    for c in df.columns:
        cl = str(c).lower()
        if cl in ["protocol", "phase", "task", "动作", "协议"]:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if "protocol" in cl or "phase" in cl or "task" in cl:
            return c
    return None


def normalize_protocol(x):
    s = str(x).strip().lower()
    aliases = {
        "restpressure": "restpressure",
        "rest pressure": "restpressure",
        "rest_pressure": "restpressure",
        "rair": "rair",
        "contraction": "contraction",
        "cough": "cough",
        "defecation": "defecation",
    }
    return aliases.get(s, s)


def composite_keys(df, patient_col, protocol_col, path_col):
    pid = df[patient_col].astype(str).str.strip()
    proto = df[protocol_col].map(normalize_protocol)
    base = df[path_col].map(basename_key)
    stem = df[path_col].map(stem_key)

    return (
        pid + "|" + proto + "|" + base,
        pid + "|" + proto + "|" + stem,
    )


def compare_keys(meta, mani, meta_col, mani_col, label):
    meta_keys = meta[meta_col].map(label)
    mani_keys = mani[mani_col].map(label)

    meta_set = set(meta_keys[meta_keys != ""])
    mani_set = set(mani_keys[mani_keys != ""])

    hit = mani_keys.isin(meta_set).sum()
    print(f"    {meta_col}  <->  {mani_col}: {hit}/{len(mani)} = {hit / max(len(mani), 1):.4f}")


def main():
    print("CACHE_DIR:", CACHE_DIR)
    print("IMAGE_EMB exists:", IMAGE_EMB.exists())
    print("IMAGE_META exists:", IMAGE_META.exists())

    emb = np.load(IMAGE_EMB, mmap_mode="r")
    meta = read_csv_smart(IMAGE_META)

    print("\n[cache]")
    print("image_embeddings shape:", emb.shape, "dtype:", emb.dtype)
    print("image_meta shape:", meta.shape)
    print("image_meta columns:", list(meta.columns))
    print(meta.head(5).to_string(index=False))

    if len(meta) != emb.shape[0]:
        print("\n[警告] image_meta 行数 != image_embeddings 第一维")
        print("len(image_meta):", len(meta))
        print("emb.shape[0]:", emb.shape[0])
    else:
        print("\n[OK] image_meta 行数和 image_embeddings 第一维一致")

    print("\n[寻找 M1/topk/manifest CSV 候选]")
    cands = find_manifest_candidates(PROJECT_DIR)
    for i, p in enumerate(cands[:30]):
        print(f"{i}: {p}")

    if not cands:
        print("\n没有找到候选 manifest。请手动把 M1 top-k4 manifest 路径填进脚本。")
        return

    print("\n[逐个候选检查对齐]")
    meta_path_cols = guess_path_cols(meta)
    meta_pid_col = guess_patient_col(meta)
    meta_proto_col = guess_protocol_col(meta)

    print("meta path cols:", meta_path_cols)
    print("meta patient col:", meta_pid_col)
    print("meta protocol col:", meta_proto_col)

    for p in cands[:20]:
        try:
            mani = read_csv_smart(p)
        except Exception as e:
            print("\n读取失败:", p, e)
            continue

        print("\n==========")
        print("candidate:", p)
        print("shape:", mani.shape)
        print("columns:", list(mani.columns))
        print(mani.head(3).to_string(index=False))

        mani_path_cols = guess_path_cols(mani)
        mani_pid_col = guess_patient_col(mani)
        mani_proto_col = guess_protocol_col(mani)

        print("manifest path cols:", mani_path_cols)
        print("manifest patient col:", mani_pid_col)
        print("manifest protocol col:", mani_proto_col)

        print("\n  basename 对齐:")
        for mc in meta_path_cols:
            for xc in mani_path_cols:
                compare_keys(meta, mani, mc, xc, basename_key)

        print("\n  stem 对齐:")
        for mc in meta_path_cols:
            for xc in mani_path_cols:
                compare_keys(meta, mani, mc, xc, stem_key)

        if meta_pid_col and meta_proto_col and mani_pid_col and mani_proto_col:
            print("\n  patient_id + protocol + basename/stem 对齐:")

            for mc in meta_path_cols:
                for xc in mani_path_cols:
                    meta_base, meta_stem = composite_keys(meta, meta_pid_col, meta_proto_col, mc)
                    mani_base, mani_stem = composite_keys(mani, mani_pid_col, mani_proto_col, xc)

                    meta_base_set = set(meta_base[meta_base != ""])
                    meta_stem_set = set(meta_stem[meta_stem != ""])

                    hit_base = mani_base.isin(meta_base_set).sum()
                    hit_stem = mani_stem.isin(meta_stem_set).sum()

                    print(f"    base {mc} <-> {xc}: {hit_base}/{len(mani)} = {hit_base / max(len(mani), 1):.4f}")
                    print(f"    stem {mc} <-> {xc}: {hit_stem}/{len(mani)} = {hit_stem / max(len(mani), 1):.4f}")

        if len(mani) == 15509:
            print("\n  [提示] 这个候选长度是 15509，很可能就是你刚才检查的 M1 top-k4 manifest。")


if __name__ == "__main__":
    main()
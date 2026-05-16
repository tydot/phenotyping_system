# sctripts/inspect_m1_frame_features.py
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = Path(r"C:\Users\Lenovo\Desktop\phenotyping_system-main\outputs\vlm\m1_topk4_vlm_manifest.csv")


def main():
    df = pd.read_csv(MANIFEST, encoding="utf-8-sig")

    print("manifest 行数:", len(df))
    print("患者数:", df["patient_id"].nunique())
    print("协议分布:")
    print(df["protocol"].value_counts())

    print("\n检查前 20 个 npy：")
    for i, row in df.head(20).iterrows():
        p = Path(str(row["feature_path_resolved"]))
        arr = np.load(p, allow_pickle=True)
        print(i, row["patient_id"], row["protocol"], row["rank"], p.name, arr.shape, arr.dtype)

    print("\n全局抽样检查 shape：")
    sample = df.sample(min(100, len(df)), random_state=0)
    shapes = {}
    for _, row in sample.iterrows():
        p = Path(str(row["feature_path_resolved"]))
        arr = np.load(p, allow_pickle=True)
        shapes[str(arr.shape)] = shapes.get(str(arr.shape), 0) + 1

    print(shapes)


if __name__ == "__main__":
    main()
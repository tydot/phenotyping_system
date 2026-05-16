# -*- coding: utf-8 -*-
"""
检查 M1-M5 merged_clinical_all.csv 是否已经具备：
1. 患者ID
2. 性别
3. consensus_cluster
4. confidence
5. 医院阈值判定所需的10个临床指标
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"H:\windows\图像数据\dataProcess\processed")

VERSIONS = {
    "M1": BASE_DIR / "M1" / "merged_clinical_all.csv",
    "M2": BASE_DIR / "M2" / "merged_clinical_all.csv",
    "M3": BASE_DIR / "M3" / "merged_clinical_all.csv",
    "M4": BASE_DIR / "M4" / "merged_clinical_all.csv",
    "M5": BASE_DIR / "M5" / "merged_clinical_all.csv",
}

REQUIRED_COLS = [
    "性别",
    "consensus_cluster",
    "confidence",
    "肛门括约肌静息压(mmHg)",
    "最大缩榨压MSP（mmHg）",
    "最大容量感觉阈值(ml)",
    "肛门括约肌长度(cm)",
    "缩肛持续时间(s)",
    "排便时直肠压力(mmHg)",
    "RAIR诱发最小容积(ml)",
    "初始感觉阈值(ml)",
    "初始便意阈值(ml)",
    "排便窘迫感阈值(ml)",
]

def main():
    print("=" * 80)
    print("检查 M1-M5 合并文件是否支持医院阈值判定")
    print("=" * 80)

    for name, path in VERSIONS.items():
        print(f"\n▶ {name}")
        print(f"路径：{path}")

        if not path.exists():
            print("❌ 文件不存在")
            continue

        df = pd.read_csv(path)
        print(f"样本数：{len(df)}")
        print(f"字段数：{len(df.columns)}")

        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

        if missing_cols:
            print("❌ 缺失字段：")
            for c in missing_cols:
                print(f"  - {c}")
        else:
            print("✅ 必要字段齐全")

        if "性别" in df.columns:
            print("\n性别分布：")
            print(df["性别"].value_counts(dropna=False))

        if "confidence" in df.columns:
            print("\nconfidence 概况：")
            print(df["confidence"].describe())

        if "consensus_cluster" in df.columns:
            print("\nCluster 分布：")
            print(df["consensus_cluster"].value_counts(dropna=False).sort_index())

        print("-" * 80)

if __name__ == "__main__":
    main()
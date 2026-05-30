"""
rebuild_rome_analysis.py

为 M1-M5 重新生成 Rome IV 分析文件。

流程：
1. Rome IV 四分型（从临床表，所有版本共用）
2. Rome 三分类映射
3. 与各版本共识标签合并 → integrated_rome_with_conf.csv
4. Rome vs Consensus 交叉表 → rome_vs_consensus_all.csv / stable.csv
5. Rome 类别内 Kruskal-Wallis 检验

用法：python scripts/rebuild_rome_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
CLINICAL_PATH = Path(r"H:\windows\image_data\dataProcess\report_valid_only_clean.csv")
OUTPUT_DIR = ROOT_DIR / "processed"

VERSIONS = ["M1", "M2", "M3", "M4", "M5"]
CONFIDENCE_THRESHOLD = 0.8


# =====================================================
# Rome IV Classifier
# =====================================================

class RomeIVClassifier:
    def __init__(self):
        self.RECTAL_PRESSURE_THRESHOLD = 45.0
        self.MSP_MRP_RATIO_LOW = 2.0
        self.MSP_MRP_RATIO_HIGH = 6.0
        self.MIN_SQUEEZE_DURATION = 3.0

    def _adequate_propulsion(self, row):
        drp = row.get('排便时直肠压力(mmHg)')
        if pd.isna(drp):
            return np.nan
        return drp >= self.RECTAL_PRESSURE_THRESHOLD

    def _dyssynergic_defecation(self, row):
        mrp = row.get('肛门括约肌静息压(mmHg)')
        msp = row.get('最大缩榨压MSP（mmHg）')
        duration = row.get('缩肛持续时间(s)')
        flags = []
        if pd.notna(mrp) and mrp > 0 and pd.notna(msp):
            ratio = msp / mrp
            if ratio < self.MSP_MRP_RATIO_LOW or ratio > self.MSP_MRP_RATIO_HIGH:
                flags.append(True)
            else:
                flags.append(False)
        else:
            flags.append(False)
        if pd.notna(duration) and duration < self.MIN_SQUEEZE_DURATION:
            flags.append(True)
        else:
            flags.append(False)
        return any(flags)

    def classify_row(self, row):
        try:
            propulsion = self._adequate_propulsion(row)
            dyssynergia = self._dyssynergic_defecation(row)
            if pd.isna(propulsion):
                return 'Data_Incomplete'
            if propulsion and dyssynergia:
                return 'Rome_Type_I'
            elif (not propulsion) and dyssynergia:
                return 'Rome_Type_II'
            elif propulsion and (not dyssynergia):
                return 'Rome_Type_III'
            elif (not propulsion) and (not dyssynergia):
                return 'Rome_Type_IV'
            else:
                return 'Unclassified'
        except Exception:
            return 'Data_Incomplete'

    def classify_dataframe(self, df):
        df = df.copy()
        df['rome_iv_type'] = df.apply(self.classify_row, axis=1)
        return df


def map_rome_iv_to_three_class(rome_iv_type):
    if rome_iv_type in ['Rome_Type_I', 'Rome_Type_III']:
        return 'Dyssynergic'
    elif rome_iv_type in ['Rome_Type_II', 'Rome_Type_IV']:
        return 'Poor_Propulsion'
    else:
        return 'Other'


# =====================================================
# Rome vs Consensus 分析
# =====================================================

def run_rome_vs_consensus(df_merge, tag, output_path):
    df_valid = df_merge[df_merge['rome_three_class'].isin(['Dyssynergic', 'Poor_Propulsion'])].copy()

    if df_valid.empty:
        print(f"  {tag}: 无有效患者，跳过")
        return

    ct = pd.crosstab(df_valid['consensus_cluster'], df_valid['rome_three_class'])

    le = LabelEncoder()
    rome_encoded = le.fit_transform(df_valid['rome_three_class'])
    cluster_encoded = le.fit_transform(df_valid['consensus_cluster'])
    ari = adjusted_rand_score(rome_encoded, cluster_encoded)
    kappa = cohen_kappa_score(rome_encoded, cluster_encoded)

    print(f"  {tag}: {len(df_valid)} patients, ARI={ari:.3f}, Kappa={kappa:.3f}")

    df_valid.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_kruskal_within_rome(df, tag, output_path):
    ROME_COL = 'rome_three_class'
    CLUSTER_COL = 'consensus_cluster'
    rome_classes = ['Dyssynergic', 'Poor_Propulsion']
    metrics = ['肛门括约肌静息压(mmHg)', '最大缩榨压MSP（mmHg）', '排便时直肠压力(mmHg)',
               '初始感觉阈值(ml)', '最大容量感觉阈值(ml)']

    results = []
    for rome in rome_classes:
        df_rome = df[df[ROME_COL] == rome]
        if df_rome.empty:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            groups = []
            valid = True
            for c in sorted(df_rome[CLUSTER_COL].dropna().unique()):
                vals = df_rome[df_rome[CLUSTER_COL] == c][metric].dropna()
                if len(vals) < 5:
                    valid = False
                    break
                groups.append(vals)
            if not valid or len(groups) < 2:
                continue
            H, p = kruskal(*groups)
            results.append({
                '分析对象': tag, 'Rome类别': rome, '指标': metric,
                'H统计量': round(H, 3), 'p值': p,
                '是否显著(p<0.05)': '是' if p < 0.05 else '否'
            })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')


# =====================================================
# 主流程
# =====================================================

def main():
    print("=" * 60)
    print("  重新生成 M1-M5 Rome IV 分析")
    print("=" * 60)

    # --- 1. Rome IV 分类（共用）---
    print("\n[1/2] Rome IV 分类...")
    clinical = pd.read_csv(CLINICAL_PATH, encoding='utf-8-sig')
    clinical['_pid_norm'] = clinical['_pid_norm'].astype(str).str.replace(r'\.0$', '', regex=True)

    classifier = RomeIVClassifier()
    clinical = classifier.classify_dataframe(clinical)
    clinical['rome_three_class'] = clinical['rome_iv_type'].apply(map_rome_iv_to_three_class)

    print(f"  Rome IV 分型: {clinical['rome_iv_type'].value_counts().to_dict()}")
    print(f"  Rome 三分类: {clinical['rome_three_class'].value_counts().to_dict()}")

    # --- 2. 各版本分析 ---
    print("\n[2/2] 各版本 Rome 分析...")

    for ver in VERSIONS:
        print(f"\n  === {ver} ===")
        ver_dir = OUTPUT_DIR / ver / "rome_analysis"
        ver_dir.mkdir(parents=True, exist_ok=True)

        # 加载共识标签
        consensus = pd.read_csv(OUTPUT_DIR / ver / "consensus_labels.csv")
        consensus['patient_id'] = consensus['patient_id'].astype(str)

        # 保存 Rome 分类文件（共用）
        clinical.to_csv(ver_dir / "report_valid_only_clean_with_romeIV.csv", index=False, encoding='utf-8-sig')
        clinical.to_csv(ver_dir / "report_valid_only_clean_with_romeIV_3class.csv", index=False, encoding='utf-8-sig')

        # 合并 Rome + Consensus
        rome_sub = clinical[['_pid_norm', 'rome_iv_type', 'rome_three_class',
                             '肛门括约肌静息压(mmHg)', '最大缩榨压MSP（mmHg）',
                             '排便时直肠压力(mmHg)', '初始感觉阈值(ml)', '最大容量感觉阈值(ml)']].copy()
        rome_sub = rome_sub.rename(columns={'_pid_norm': 'patient_id'})

        consensus_sub = consensus[['patient_id', 'consensus_cluster', 'confidence']].copy()

        integrated = pd.merge(rome_sub, consensus_sub, on='patient_id', how='inner')
        integrated.to_csv(ver_dir / "integrated_rome_with_conf.csv", index=False, encoding='utf-8-sig')

        # Rome vs Consensus
        run_rome_vs_consensus(integrated, "all", ver_dir / "rome_vs_consensus_all.csv")

        stable = integrated[integrated['confidence'] >= CONFIDENCE_THRESHOLD].copy()
        run_rome_vs_consensus(stable, "stable", ver_dir / "rome_vs_consensus_stable.csv")

        # Kruskal within Rome
        run_kruskal_within_rome(integrated, "all", ver_dir / "kruskal_within_rome_all.csv")
        run_kruskal_within_rome(stable, "stable", ver_dir / "kruskal_within_rome_stable.csv")

    print("\n完成!")


if __name__ == "__main__":
    main()

"""
rebuild_umap.py

为 M1-M5 重新生成 3D UMAP 可视化图。
从嵌入生成 UMAP，与新共识标签对齐后着色。

用法：python scripts/rebuild_umap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "processed"

# 嵌入和 seed0 路径
EMBED_SOURCES = {
    "M1": {
        "embed": Path(r"H:\windows\image_data\dataProcess\outputs\fair_compare_multi_dim\attn\topk4\pca50\patient_embeddings_concat.npy"),
        "ids": Path(r"H:\windows\image_data\dataProcess\outputs\fair_compare_multi_dim\attn\topk4\pca50\consensus_labels.csv"),
    },
    "M2": {
        "embed": Path(r"H:\windows\image_data\dataProcess\outputs\fair_compare_multi_dim\attn\topk6\pca50\patient_embeddings_concat.npy"),
        "ids": Path(r"H:\windows\image_data\dataProcess\outputs\fair_compare_multi_dim\attn\topk6\pca50\consensus_labels_all.csv"),
    },
    "M3": {
        "embed": Path(r"H:\windows\image_data\dataProcess\outputs\fair_compare_multi_dim\mean\topk0\pca50\patient_embeddings_concat.npy"),
        "ids": Path(r"H:\windows\image_data\dataProcess\outputs\fair_compare_multi_dim\mean\topk0\pca50\consensus_labels.csv"),
    },
    "M4": {
        "embed": Path(r"H:\windows\image_data\dataProcess\rerun_outputs_02468_pca50\attn\topk4\seed0\patient_embeddings_concat.npy"),
        "ids": Path(r"H:\windows\image_data\dataProcess\rerun_outputs_02468_pca50\attn\topk4\seed0\clusters.csv"),
    },
    "M5": {
        "embed": Path(r"H:\windows\image_data\dataProcess\rerun_outputs_02468_pca50\mean\mapp6\patient_embeddings_consensus.npy"),
        "ids": Path(r"H:\windows\image_data\dataProcess\rerun_outputs_02468_pca50\mean\mapp6\seed0\clusters.csv"),
    },
}

CONFIDENCE_THRESHOLD = 0.8
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']
BOUNDARY_COLOR = 'lightgray'


def normalize_pid(x):
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def generate_umap_3d(embeddings):
    """从嵌入生成 3D UMAP 坐标。"""
    from umap import UMAP
    reducer = UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def plot_umap_3d(df, title, output_path):
    """绘制 3D UMAP 图。"""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    stable = df[df['confidence'] >= CONFIDENCE_THRESHOLD]
    for c in sorted(stable['consensus_cluster'].unique()):
        sub = stable[stable['consensus_cluster'] == c]
        ax.scatter(
            sub['UMAP_3D_X'], sub['UMAP_3D_Y'], sub['UMAP_3D_Z'],
            s=8, alpha=0.8, color=COLORS[int(c) % len(COLORS)],
            label=f'Cluster {int(c)}'
        )

    boundary = df[df['confidence'] < CONFIDENCE_THRESHOLD]
    if len(boundary) > 0:
        ax.scatter(
            boundary['UMAP_3D_X'], boundary['UMAP_3D_Y'], boundary['UMAP_3D_Z'],
            s=8, alpha=0.4, color=BOUNDARY_COLOR,
            label=f'Boundary (conf < {CONFIDENCE_THRESHOLD})'
        )

    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_zlabel('UMAP-3')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("  重新生成 M1-M5 UMAP 图")
    print("=" * 60)

    for ver, sources in EMBED_SOURCES.items():
        print(f"\n  === {ver} ===")

        # 加载嵌入
        embeddings = np.load(sources["embed"])
        print(f"  嵌入: {embeddings.shape}")

        # 加载患者 ID
        ids_df = pd.read_csv(sources["ids"])
        ids_df['patient_id'] = ids_df['patient_id'].astype(str).apply(normalize_pid)
        patient_ids = ids_df['patient_id'].values

        # 加载新共识标签
        consensus = pd.read_csv(OUTPUT_DIR / ver / "consensus_labels.csv")
        consensus['patient_id'] = consensus['patient_id'].astype(str).apply(normalize_pid)
        consensus_pids = set(consensus['patient_id'])

        # 对齐到 1067 患者
        mask = np.array([pid in consensus_pids for pid in patient_ids])
        embeddings_filtered = embeddings[mask]
        pids_filtered = patient_ids[mask]

        print(f"  对齐后: {len(pids_filtered)} 患者")

        # 生成 3D UMAP
        print(f"  生成 UMAP...")
        coords = generate_umap_3d(embeddings_filtered)

        # 构建 DataFrame
        umap_df = pd.DataFrame({
            'patient_id': pids_filtered,
            'UMAP_3D_X': coords[:, 0],
            'UMAP_3D_Y': coords[:, 1],
            'UMAP_3D_Z': coords[:, 2],
        })

        # 合并共识标签
        umap_df = umap_df.merge(
            consensus[['patient_id', 'consensus_cluster', 'confidence']],
            on='patient_id',
            how='inner'
        )

        # 绘图
        out_dir = OUTPUT_DIR / ver
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{ver} - Patient-level 3D UMAP with Consensus Clusters"
        output_path = out_dir / "umap_consensus_clusters.png"
        plot_umap_3d(umap_df, title, output_path)
        print(f"  保存: {output_path}")

    print("\n完成!")


if __name__ == "__main__":
    main()

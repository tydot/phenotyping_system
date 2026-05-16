"""
Mean Pooling 表征生成器
使用 MAPP=6 采样
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data_loader.npy_loader import NPYFeatureLoader


class PatientEmbedder:
    """
    基于 Mean Pooling 的患者表征生成器

    配置：
    - MAPP=6（每协议最多随机采样 6 张图像）
    - seed=42（保证可复现）
    """

    PROTOCOL_ORDER = ['RestPressure', 'Contraction', 'Defecation', 'Cough', 'rair']

    def __init__(
        self,
        features_dir: str,
        mapp: int = 6,
        pooling_method: str = 'mean',
        seed: int = 42
    ):
        """
        Args:
            features_dir: DINOv2 特征目录
            mapp: MAPP 采样参数（默认 6）
            pooling_method: 池化方法（'mean' 或 'max'）
            seed: 随机种子
        """
        self.loader = NPYFeatureLoader(
            features_dir=features_dir,
            mapp=mapp,
            seed=seed
        )
        self.pooling_method = pooling_method
        self.mapp = mapp
        self.seed = seed

    def aggregate_protocol(
        self,
        protocol_features: np.ndarray
    ) -> np.ndarray:
        """协议级聚合（Mean Pooling）"""
        if self.pooling_method == 'mean':
            return np.mean(protocol_features, axis=0)
        elif self.pooling_method == 'max':
            return np.max(protocol_features, axis=0)
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")

    def generate_patient_embedding(
        self,
        patient_id: str,
        check_completeness: bool = True
    ) -> np.ndarray:
        """
        生成单个患者的表征向量（Mean Pooling）

        Returns:
            患者向量 (3840,) = 5 × 768
        """
        protocols = self.loader.load_patient(patient_id)

        if check_completeness:
            completeness = self.loader.check_completeness(protocols)
            missing = [p for p, valid in completeness.items() if not valid]

            if missing:
                raise ValueError(
                    f"患者 {patient_id} 缺失协议: {missing}"
                )

        protocol_vectors = []

        for protocol_name in self.PROTOCOL_ORDER:
            if protocol_name not in protocols:
                raise ValueError(f"患者 {patient_id} 缺失协议: {protocol_name}")

            protocol_vector = self.aggregate_protocol(
                protocols[protocol_name]
            )

            protocol_vectors.append(protocol_vector)

        patient_vector = np.concatenate(protocol_vectors)

        return patient_vector

    def batch_generate(
        self,
        patient_ids: List[str],
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """批量生成患者表征"""
        embeddings = {}

        for patient_id in patient_ids:
            if verbose:
                print(f"🔄 生成患者 {patient_id} 的表征（Mean Pooling, MAPP={self.mapp})...")

            try:
                embedding = self.generate_patient_embedding(patient_id)
                embeddings[patient_id] = embedding

                if verbose:
                    print(f"  ✅ 维度: {embedding.shape}")

            except Exception as e:
                print(f"  ❌ 失败: {e}")

        return embeddings

    def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        output_path: str
    ):
        """保存患者表征"""
        patient_ids = list(embeddings.keys())
        vectors = np.array([embeddings[pid] for pid in patient_ids])

        np.savez(
            output_path,
            patient_ids=patient_ids,
            embeddings=vectors,
            protocol_order=self.PROTOCOL_ORDER,
            pooling_method=self.pooling_method,
            mapp=self.mapp,
            seed=self.seed
        )

        print(f"💾 已保存 {len(patient_ids)} 个患者表征（Mean）到: {output_path}")

    @staticmethod
    def load_embeddings(npz_path: str) -> Dict[str, np.ndarray]:
        """从文件加载患者表征"""
        data = np.load(npz_path)

        patient_ids = data['patient_ids']
        vectors = data['embeddings']

        embeddings = {
            pid: vec for pid, vec in zip(patient_ids, vectors)
        }

        print(f"📂 已加载 {len(embeddings)} 个患者表征（Mean Pooling）")
        print(f"   MAPP={int(data['mapp'])}, seed={int(data['seed'])}")

        return embeddings

"""
Attention Pooling 表征生成器
使用完整数据 + Top-k 选择
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data_loader.npy_loader import NPYFeatureLoader


class AttentionPooling:
    """
    基于质心的确定性注意力池化

    核心思想：
    1. 计算协议内所有帧的质心
    2. 计算每帧与质心的余弦相似度
    3. 选择 top-k 个高分帧
    4. Softmax 归一化得到注意力权重
    5. 加权求和得到协议级向量
    """

    def __init__(self, temperature: float = 0.07, topk: Optional[int] = None):
        self.temperature = temperature
        self.topk = topk

    @staticmethod
    def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """L2 归一化"""
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax"""
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

    def aggregate(
            self,
            protocol_features: np.ndarray,
            return_weights: bool = False
    ) -> np.ndarray:
        """协议级注意力池化"""
        n_frames, dim = protocol_features.shape

        if n_frames == 1:
            if return_weights:
                return protocol_features[0], np.array([1.0])
            return protocol_features[0]

        # L2 归一化
        E_norm = self._l2norm(protocol_features, axis=1)

        # 计算质心
        centroid = self._l2norm(E_norm.mean(axis=0, keepdims=True)).squeeze(0)

        # 余弦相似度
        scores = (E_norm @ centroid).astype(np.float32)

        # Top-k 选择
        if self.topk is not None and n_frames > self.topk:
            topk_indices = np.argsort(-scores)[:self.topk]
            selected_features = protocol_features[topk_indices]
            selected_scores = scores[topk_indices]
        else:
            selected_features = protocol_features
            selected_scores = scores

        # Softmax 权重
        scaled_scores = selected_scores / self.temperature
        attention_weights = self._softmax(scaled_scores.reshape(1, -1), axis=1).reshape(-1)

        # 加权求和
        aggregated = (attention_weights[:, None] * selected_features).sum(axis=0)

        if return_weights:
            return aggregated, attention_weights

        return aggregated


class AttentionEmbedder:
    """
    基于 Attention Pooling 的患者表征生成器

    配置：
    - MAPP=None（加载所有图像）
    - topk=8（选择最相似的 8 张图像）
    - temperature=0.07（Softmax 温度）
    """

    PROTOCOL_ORDER = ['RestPressure', 'Contraction', 'Defecation', 'Cough', 'rair']

    def __init__(
            self,
            features_dir: str,
            mapp: Optional[int] = None,
            temperature: float = 0.07,
            topk: Optional[int] = 8,
            seed: Optional[int] = None
    ):
        """
        Args:
            features_dir: DINOv2 特征目录
            mapp: MAPP 采样（None = 加载所有图像）
            temperature: Softmax 温度（τ=0.07 最优）
            topk: Top-k 选择（8 最优）
            seed: 随机种子（Attention 是确定性的，通常设为 None）
        """
        self.loader = NPYFeatureLoader(
            features_dir=features_dir,
            mapp=mapp,
            seed=seed
        )
        self.attention_pooling = AttentionPooling(
            temperature=temperature,
            topk=topk
        )
        self.temperature = temperature
        self.topk = topk

    def generate_patient_embedding(
            self,
            patient_id: str,
            check_completeness: bool = True
    ) -> np.ndarray:
        """生成单个患者的表征向量（Attention Pooling）"""
        protocols = self.loader.load_patient(patient_id)

        if check_completeness:
            completeness = self.loader.check_completeness(protocols)
            missing = [p for p, valid in completeness.items() if not valid]

            if missing:
                raise ValueError(f"患者 {patient_id} 缺失协议: {missing}")

        protocol_vectors = []

        for protocol_name in self.PROTOCOL_ORDER:
            if protocol_name not in protocols:
                raise ValueError(f"患者 {patient_id} 缺失协议: {protocol_name}")

            protocol_vector = self.attention_pooling.aggregate(
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
                print(f"🔄 生成患者 {patient_id} 的表征（Attention, topk={self.topk}, τ={self.temperature})...")

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
            pooling_method='attention',
            temperature=self.temperature,
            topk=self.topk if self.topk else -1
        )

        print(f"💾 已保存 {len(patient_ids)} 个患者表征（Attention）到: {output_path}")

    @staticmethod
    def load_embeddings(npz_path: str) -> Dict[str, np.ndarray]:
        """从文件加载患者表征"""
        data = np.load(npz_path)

        patient_ids = data['patient_ids']
        vectors = data['embeddings']

        embeddings = {
            pid: vec for pid, vec in zip(patient_ids, vectors)
        }

        print(f"📂 已加载 {len(embeddings)} 个患者表征（Attention Pooling）")
        print(f"   τ={float(data['temperature'])}, top-k={int(data['topk'])}")

        return embeddings

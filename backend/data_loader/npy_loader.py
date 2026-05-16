"""
NPY 特征加载器（生产版本 v2.0）

数据格式：
- 每个 .npy 文件固定 (3, 768) shape
- 每个协议包含 1-5 个 .npy 文件
- 总帧数范围：3-15 帧

设计：
1. MAPP 采样：按时间顺序取前 N 帧（确定性）
2. 严格模式：协议缺失时抛出异常
3. 时序保留：返回 frame_ids 用于后续建模

作者：[你的名字]
日期：2024-12-XX
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import warnings


class NPYFeatureLoader:
    """
    DINOv2 特征加载器（已针对实际数据格式优化）

    数据格式说明：
        - 每个 .npy 文件包含固定 3 帧特征，shape=(3, 768)
        - 每个协议包含多个 .npy 文件（1-5 个不等）
        - 需要将所有文件的帧 vstack 合并

    示例：
        Contraction 协议有 4 个文件：
        - file-1.npy: (3, 768)
        - file-2.npy: (3, 768)  → vstack → (12, 768)
        - file-3.npy: (3, 768)
        - file-4.npy: (3, 768)
    """

    PROTOCOL_ORDER = ['RestPressure', 'Contraction', 'Defecation', 'Cough', 'rair']

    def __init__(
            self,
            features_dir: str,
            mapp: Optional[int] = None,
            seed: Optional[int] = None
    ):
        """
        Args:
            features_dir: DINOv2 特征根目录
            mapp: MAPP 采样参数
                - None: 加载所有帧（Attention Pooling）
                - int: 按时间顺序取前 N 帧（Mean Pooling）
            seed: 已废弃（保留向后兼容）
        """
        self.features_dir = Path(features_dir)
        self.mapp = mapp

        if seed is not None:
            warnings.warn(
                "参数 'seed' 已废弃。系统使用确定性时序采样。",
                DeprecationWarning,
                stacklevel=2
            )

        if not self.features_dir.exists():
            raise FileNotFoundError(f"特征目录不存在: {features_dir}")

    def load_protocol(
            self,
            patient_id: str,
            protocol: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个协议的所有帧（多文件自动合并）

        Args:
            patient_id: 患者ID
            protocol: 协议名称

        Returns:
            (features, frame_ids)
                - features: (N, 768) 合并后的特征矩阵
                - frame_ids: (N,) 帧序列号

        处理逻辑：
            1. 加载所有 .npy 文件
            2. 验证每个文件的 shape = (3, 768)
            3. vstack 合并所有帧
            4. 如果设置了 MAPP，取前 mapp 帧
        """
        protocol_dir = self.features_dir / patient_id / protocol

        if not protocol_dir.exists():
            raise FileNotFoundError(f"协议目录不存在: {protocol_dir}")

        # 加载所有 .npy 文件（按文件名排序保证时间顺序）
        npy_files = sorted(protocol_dir.glob("*.npy"))

        if len(npy_files) == 0:
            raise ValueError(f"协议 {protocol} 没有 .npy 文件: {protocol_dir}")

        # 收集所有帧
        all_frames = []

        for npy_file in npy_files:
            feat = np.load(npy_file)

            # 验证格式：应为 (3, 768)
            if feat.shape != (3, 768):
                # 如果是 (768,) 或 (1, 768)，尝试修正
                if feat.ndim == 1 and feat.shape[0] == 768:
                    feat = feat.reshape(1, -1)
                elif feat.ndim == 2 and feat.shape == (1, 768):
                    pass  # 已经是正确格式
                elif feat.ndim == 2 and feat.shape[1] == 768:
                    # 形如 (N, 768)，接受
                    pass
                else:
                    raise ValueError(
                        f"特征格式错误: {npy_file}\n"
                        f"期望 (3, 768) 或 (N, 768)，得到 {feat.shape}"
                    )

            all_frames.append(feat)

        # 合并所有帧：[(3,768), (3,768), ...] → (N, 768)
        all_features = np.vstack(all_frames)
        frame_ids = np.arange(len(all_features))

        # MAPP 采样（确定性时序截断）
        if self.mapp is not None and len(all_features) > self.mapp:
            all_features = all_features[:self.mapp]
            frame_ids = frame_ids[:self.mapp]

        return all_features, frame_ids

    def load_patient(
            self,
            patient_id: str,
            strict: bool = True,
            return_frame_ids: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        加载单个患者的所有协议

        Args:
            patient_id: 患者ID
            strict: 严格模式（协议缺失时抛出异常）
            return_frame_ids: 是否返回时序信息

        Returns:
            strict=True 时保证返回 5 个完整协议
            strict=False 时返回可用协议（可能 < 5）
        """
        protocols = {}
        frame_ids_dict = {}

        for protocol_name in self.PROTOCOL_ORDER:
            try:
                features, frame_ids = self.load_protocol(patient_id, protocol_name)
                protocols[protocol_name] = features
                frame_ids_dict[protocol_name] = frame_ids

            except Exception as e:
                if strict:
                    raise ValueError(
                        f"患者 {patient_id} 协议 {protocol_name} 加载失败（严格模式）: {e}"
                    ) from e
                else:
                    warnings.warn(
                        f"患者 {patient_id} 协议 {protocol_name} 已跳过: {e}",
                        stacklevel=2
                    )

        # 严格模式：检查协议完整性
        if strict and len(protocols) < len(self.PROTOCOL_ORDER):
            missing = set(self.PROTOCOL_ORDER) - set(protocols.keys())
            raise ValueError(
                f"患者 {patient_id} 协议不完整。缺失: {missing}"
            )

        if return_frame_ids:
            return {
                "features": protocols,
                "frame_ids": frame_ids_dict
            }

        return protocols

    def check_completeness(
            self,
            protocols: Dict[str, np.ndarray]
    ) -> Dict[str, bool]:
        """检查协议完整性"""
        return {
            protocol: protocol in protocols
            for protocol in self.PROTOCOL_ORDER
        }

    def get_protocol_stats(
            self,
            patient_id: str
    ) -> Dict[str, Dict[str, int]]:
        """
        获取协议统计（用于诊断）

        Returns:
            {
                protocol_name: {
                    "total_frames": 原始总帧数,
                    "loaded_frames": MAPP 后帧数,
                    "n_files": 文件数,
                    "mapp_limit": MAPP 限制
                },
                ...
            }
        """
        stats = {}

        for protocol_name in self.PROTOCOL_ORDER:
            try:
                protocol_dir = self.features_dir / patient_id / protocol_name
                npy_files = sorted(protocol_dir.glob("*.npy"))

                # 计算原始总帧数
                total_frames = 0
                for npy_file in npy_files:
                    feat = np.load(npy_file)
                    total_frames += feat.shape[0]

                loaded_frames = min(total_frames, self.mapp) if self.mapp else total_frames

                stats[protocol_name] = {
                    "total_frames": total_frames,
                    "loaded_frames": loaded_frames,
                    "n_files": len(npy_files),
                    "mapp_limit": self.mapp
                }
            except:
                stats[protocol_name] = {
                    "total_frames": 0,
                    "loaded_frames": 0,
                    "n_files": 0,
                    "mapp_limit": self.mapp
                }

        return stats


# ===== 部署健全性检查 =====
def deployment_sanity_check(loader: NPYFeatureLoader, test_patient_id: str):
    """
    部署前 5 项健全性检查

    Args:
        loader: 配置好的 NPYFeatureLoader
        test_patient_id: 测试用患者ID

    Raises:
        AssertionError: 任一检查失败
    """
    print("🔍 开始部署健全性检查...\n")

    # Check 1: 严格模式加载完整患者
    try:
        protocols = loader.load_patient(test_patient_id, strict=True)
        assert len(protocols) == 5, f"期望 5 个协议，得到 {len(protocols)}"
        print("  ✅ Check 1: 严格模式加载完整患者")
    except ValueError as e:
        raise AssertionError(f"Check 1 失败: {e}")

    # Check 2: 所有协议特征维度正确
    for proto_name, feat in protocols.items():
        assert feat.ndim == 2, f"{proto_name} 应为 2D 数组"
        assert feat.shape[1] == 768, f"{proto_name} 特征维度应为 768"
    print("  ✅ Check 2: 所有协议特征维度正确 (N, 768)")

    # Check 3: MAPP 限制生效
    if loader.mapp is not None:
        for proto_name, feat in protocols.items():
            assert len(feat) <= loader.mapp, \
                f"{proto_name} 帧数 {len(feat)} 超过 MAPP={loader.mapp}"
    print(f"  ✅ Check 3: MAPP 限制生效（mapp={loader.mapp}）")

    # Check 4: 时序信息可正确返回
    result = loader.load_patient(test_patient_id, return_frame_ids=True)
    assert "features" in result and "frame_ids" in result
    for proto in loader.PROTOCOL_ORDER:
        if proto in result["features"]:
            assert len(result["features"][proto]) == len(result["frame_ids"][proto])
    print("  ✅ Check 4: 时序信息可正确返回")

    # Check 5: 统计信息可正确获取
    stats = loader.get_protocol_stats(test_patient_id)
    assert len(stats) == 5
    for proto_name, stat in stats.items():
        assert "total_frames" in stat
        assert "loaded_frames" in stat
        assert "n_files" in stat
    print("  ✅ Check 5: 统计信息可正确获取")

    print("\n✅ 所有健全性检查通过！系统可以部署。\n")


# ===== 测试代码 =====
if __name__ == "__main__":
    FEATURES_DIR = r"D:\dataProcess\dinov2_features"

    print("=" * 70)
    print("🧪 NPY 加载器测试（生产版本 v2.0）")
    print("=" * 70)

    # ========== 测试1：Mean Pooling 模式 ==========
    print("\n📋 测试1：Mean Pooling 模式（MAPP=6）")
    print("-" * 70)

    loader_mean = NPYFeatureLoader(
        features_dir=FEATURES_DIR,
        mapp=6
    )

    protocols = loader_mean.load_patient("002", strict=True)

    print("协议加载结果：")
    for proto_name, feat in protocols.items():
        print(f"  {proto_name:15s}: {feat.shape}")

    # ========== 测试2：Attention Pooling 模式 ==========
    print("\n📋 测试2：Attention Pooling 模式（完整加载）")
    print("-" * 70)

    loader_attn = NPYFeatureLoader(
        features_dir=FEATURES_DIR,
        mapp=None
    )

    protocols_full = loader_attn.load_patient("002", strict=True)

    print("协议加载结果：")
    for proto_name, feat in protocols_full.items():
        print(f"  {proto_name:15s}: {feat.shape}")

    # ========== 测试3：时序信息 ==========
    print("\n📋 测试3：时序信息返回")
    print("-" * 70)

    result = loader_mean.load_patient("002", return_frame_ids=True)

    print("时序信息：")
    for proto_name in result["features"].keys():
        feat = result["features"][proto_name]
        frame_ids = result["frame_ids"][proto_name]
        print(f"  {proto_name:15s}: {len(feat)} 帧，frame_ids={list(frame_ids)}")

    # ========== 测试4：协议统计 ==========
    print("\n📋 测试4：协议统计信息")
    print("-" * 70)

    stats = loader_mean.get_protocol_stats("002")

    print("协议统计：")
    print(f"  {'协议':<15s} {'文件数':>6s} {'总帧数':>8s} {'加载帧数':>10s} {'MAPP限制':>10s}")
    print("  " + "-" * 60)
    for proto_name, stat in stats.items():
        print(f"  {proto_name:<15s} "
      f"{stat['n_files']:>6d} "
      f"{stat['total_frames']:>8d} "
      f"{stat['loaded_frames']:>10d} "
      f"{str(stat['mapp_limit']) if stat['mapp_limit'] else 'None':>10s}")  # ✅ 转为字符串


    # ========== 测试5：健全性检查 ==========
    print("\n📋 测试5：部署健全性检查")
    print("-" * 70)

    deployment_sanity_check(loader_mean, "002")

    print("=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)

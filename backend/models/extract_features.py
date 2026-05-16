"""
DINOv2特征批量提取脚本
功能：将图像数据 (N, 224, 224) 转换为 DINOv2 特征 (N, 768)
作者：周嘉琦
日期：2026-01-22
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


class DINOv2FeatureExtractor:
    """DINOv2特征提取器（批量处理版）"""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 加载 DINOv2 模型到 {self.device}...")
        # 使用 torch.hub 加载
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.to(self.device)
        self.model.eval()
        print("✅ 模型加载完成\n")
    
    def extract_from_npy(self, npy_path: Path) -> np.ndarray:
        """
        从 .npy 文件提取 DINOv2 特征
        
        Args:
            npy_path: .npy 文件路径（存储图像数据）
            
        Returns:
            DINOv2 特征 (N_frames, 768)
        """
        # 加载图像数据
        images = np.load(npy_path)  # (N, 224, 224) 或 (N, 224, 224, 3)
        
        # 检查维度
        if images.ndim == 2:  # (224, 224)
            images = images[np.newaxis, ...]  # (1, 224, 224)
        
        # 确保是 (N, 224, 224, 3) 格式
        if images.shape[-1] != 3:
            # 灰度图 → RGB
            images = np.stack([images] * 3, axis=-1)
        
        # 转为 torch 张量并归一化
        images_tensor = torch.from_numpy(images).float()  # (N, 224, 224, 3)
        images_tensor = images_tensor.permute(0, 3, 1, 2)  # (N, 3, 224, 224)
        images_tensor = images_tensor / 255.0  # [0, 1]
        
        # 标准化（ImageNet均值和方差）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_tensor = (images_tensor - mean) / std
        
        images_tensor = images_tensor.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(images_tensor)  # (N, 768)
        
        return features.cpu().numpy()
    
    def convert_patient_folder(
        self,
        patient_dir: Path,
        output_dir: Path,
        overwrite: bool = False
    ):
        """
        转换单个患者文件夹的所有 .npy 文件
        
        Args:
            patient_dir: 患者原始数据目录
            output_dir: 输出目录（保持相同结构）
            overwrite: 是否覆盖已存在的文件
        """
        patient_id = patient_dir.name
        print(f"\n📂 处理患者: {patient_id}")
        
        # 遍历所有协议文件夹
        for protocol_dir in patient_dir.iterdir():
            if not protocol_dir.is_dir():
                continue
            
            protocol_name = protocol_dir.name
            
            # 创建输出目录
            output_protocol_dir = output_dir / patient_id / protocol_name
            output_protocol_dir.mkdir(parents=True, exist_ok=True)
            
            # 处理所有 .npy 文件
            npy_files = list(protocol_dir.glob("*.npy"))
            
            print(f"  📁 {protocol_name}: {len(npy_files)} 文件")
            
            for npy_file in tqdm(npy_files, desc=f"    {protocol_name}", leave=False):
                output_file = output_protocol_dir / npy_file.name
                
                # 检查是否跳过
                if output_file.exists() and not overwrite:
                    continue
                
                try:
                    # 提取特征
                    features = self.extract_from_npy(npy_file)
                    
                    # 保存
                    np.save(output_file, features)
                    
                except Exception as e:
                    print(f"      ❌ 失败: {npy_file.name} - {e}")
    
    def batch_convert(
        self,
        input_root: Path,
        output_root: Path,
        patient_ids: list = None,
        overwrite: bool = False
    ):
        """
        批量转换多个患者
        
        Args:
            input_root: 输入根目录
            output_root: 输出根目录
            patient_ids: 患者ID列表（None则处理所有）
            overwrite: 是否覆盖已存在的文件
        """
        if patient_ids is None:
            patient_dirs = [d for d in input_root.iterdir() if d.is_dir()]
        else:
            patient_dirs = [input_root / pid for pid in patient_ids]
        
        print(f"🚀 开始批量转换 {len(patient_dirs)} 个患者")
        print(f"📥 输入目录: {input_root}")
        print(f"📤 输出目录: {output_root}")
        
        for patient_dir in patient_dirs:
            if not patient_dir.exists():
                print(f"⚠️ 跳过不存在的患者: {patient_dir.name}")
                continue
            
            self.convert_patient_folder(patient_dir, output_root, overwrite)
        
        print("\n✅ 所有患者处理完成！")


# ===== 测试代码 =====
if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = Path(r"D:\dataProcess\preprocessed_features")
    OUTPUT_DIR = Path(r"D:\dataProcess\dinov2_features")
    
    # 创建提取器
    extractor = DINOv2FeatureExtractor()
    
    # 批量转换（先测试患者002）
    extractor.batch_convert(
        input_root=INPUT_DIR,
        output_root=OUTPUT_DIR,
        patient_ids=None,  # ← None表示处理所有患者
        overwrite=False  # ← 改为False，避免重复处理
    )
    
    print("\n📊 验证转换结果:")
    test_file = OUTPUT_DIR / "002" / "Contraction" / "002-易良富-2022-6-29-Contraction(提肛收缩)-1.npy"
    if test_file.exists():
        features = np.load(test_file)
        print(f"  原始文件: (N, 224, 224)")
        print(f"  转换后: {features.shape}  ← 应该是 (N, 768)")
    else:
        print("  ❌ 未找到测试文件")

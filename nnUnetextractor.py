import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from batchgenerators.utilities.file_and_folder_operations import join, load_json
import blosc2
import warnings
warnings.filterwarnings('ignore')

# ====================== nnUNet特征提取器 ======================
class nnUNetFeatureExtractor:
    """
    简化版nnUNet特征提取器
    功能：加载nnUNet模型，提取医学影像的瓶颈层特征（320维）
    """
    
    def __init__(
        self,
        preprocessed_folder: str,
        checkpoint_path: str,
        plans_path: str,
        dataset_json_path: str,
        configuration: str = "3d_fullres",
        device: Optional[str] = None
    ):
        """
        初始化特征提取器
        
        Args:
            preprocessed_folder: 预处理数据文件夹路径
            checkpoint_path: 模型权重路径 (.pth)
            plans_path: nnUNetPlans.json路径
            dataset_json_path: dataset.json路径
            configuration: 配置名称，默认"3d_fullres"
            device: 设备，默认自动检测 (cuda/cpu)
        """
        # 设置设备
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 保存路径
        self.preprocessed_folder = preprocessed_folder
        self.checkpoint_path = checkpoint_path
        self.plans_path = plans_path
        self.dataset_json_path = dataset_json_path
        self.configuration = configuration
        
        # 初始化模型
        self._init_model()
        print(f"✅ nnUNet特征提取器初始化完成 | 设备: {self.device}")
    
    def _init_model(self):
        """初始化nnUNet模型"""
        # 1. 验证文件存在
        self._validate_files()
        
        # 2. 加载配置
        self.config_manager, self.dataset_json, self.num_input_channels, self.num_output_channels = self._load_configs()
        
        # 3. 构建网络
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        
        self.model = get_network_from_plans(
            self.config_manager.network_arch_class_name,
            self.config_manager.network_arch_init_kwargs,
            self.config_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.num_output_channels,
            allow_init=True,
            deep_supervision=False
        ).to(self.device)
        
        # 4. 加载权重
        self._load_weights()
        
        # 5. 设置为评估模式
        self.model.eval()
        
        # 6. 打印基本信息
        print(f"   模型输入通道数: {self.num_input_channels}")
        print(f"   模型输出通道数: {self.num_output_channels}")
        print(f"   特征维度: 320 (瓶颈层通道数)")
    
    def _validate_files(self):
        """验证必需文件是否存在"""
        files_to_check = [
            (self.preprocessed_folder, "预处理数据文件夹"),
            (self.checkpoint_path, "模型权重文件"),
            (self.plans_path, "plans.json文件"),
            (self.dataset_json_path, "dataset.json文件")
        ]
        
        for path, name in files_to_check:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}不存在: {path}")
    
    def _load_configs(self):
        """加载训练配置"""
        # 加载JSON文件
        plans = load_json(self.plans_path)
        dataset_json = load_json(self.dataset_json_path)
        
        # 创建PlansManager
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        plans_manager = PlansManager(plans)
        
        # 获取配置管理器
        try:
            config_manager = plans_manager.get_configuration(self.configuration)
        except KeyError:
            available_configs = list(plans_manager.plans['configurations'].keys())
            raise ValueError(f"配置 '{self.configuration}' 不存在，可用配置: {available_configs}")
        
        # 获取输入输出通道数
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
        
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_output_channels = label_manager.num_segmentation_heads
        
        return config_manager, dataset_json, num_input_channels, num_output_channels
    
    def _load_weights(self):
        """加载模型权重"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 提取权重
        if 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除module前缀（如果是多GPU训练保存的）
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            new_state_dict[new_k] = v
        
        self.model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ 模型权重加载成功: {self.checkpoint_path}")
    
    def _load_sample_data(self, sample_name: str) -> torch.Tensor:
        """加载单个样本的预处理数据"""
        data_path = join(self.preprocessed_folder, f"{sample_name}.b2nd")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 使用blosc2加载
        blosc2.set_nthreads(1)
        data_blosc2 = blosc2.open(urlpath=data_path, mode='r', dparams={'nthreads': 1})
        data_numpy = data_blosc2[:]
        
        # 调整形状: (channels, D, H, W) → (1, channels, D, H, W)
        if len(data_numpy.shape) == 4:
            data_with_batch = data_numpy[np.newaxis, ...]
        elif len(data_numpy.shape) == 5:
            data_with_batch = data_numpy[0:1, ...] if data_numpy.shape[0] > 1 else data_numpy
        else:
            raise ValueError(f"不支持的数据形状: {data_numpy.shape}")
        
        # 转换为tensor
        data_tensor = torch.from_numpy(data_with_batch.astype(np.float32)).to(self.device)
        
        return data_tensor
    
    def extract_single_feature(self, sample_name: str) -> np.ndarray:
        """
        提取单个样本的特征向量
        
        Args:
            sample_name: 样本名称（不含.b2nd扩展名）
            
        Returns:
            特征向量 (320维)
        """
        with torch.no_grad():
            # 1. 加载数据
            data_tensor = self._load_sample_data(sample_name)
            
            # 2. 验证通道数
            if data_tensor.shape[1] != self.num_input_channels:
                print(f"⚠️  警告: 数据通道数({data_tensor.shape[1]})与模型输入通道数({self.num_input_channels})不匹配")
            
            # 3. 提取编码器输出
            encoder_outputs = self.model.encoder(data_tensor)
            
            # 4. 获取瓶颈层特征（最后一个编码器输出）
            if isinstance(encoder_outputs, (list, tuple)):
                bottleneck_feature = encoder_outputs[-1]  # 形状: (1, 320, D, H, W)
            else:
                bottleneck_feature = encoder_outputs
            
            # 5. 全局平均池化
            # dim=[2,3,4]对应(D, H, W)空间维度
            global_feature = torch.mean(bottleneck_feature, dim=[2, 3, 4])  # 形状: (1, 320)
            
            # 6. 展平为一维数组
            feature_vector = global_feature.cpu().numpy().flatten()
            
            return feature_vector
    
    def extract_batch_features(self, sample_names: List[str], output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        批量提取特征
        
        Args:
            sample_names: 样本名称列表
            output_csv: 输出CSV文件路径（可选）
            
        Returns:
            包含所有特征的DataFrame，列: ['sample_name', 'feature_0', 'feature_1', ..., 'feature_319']
        """
        features_data = []
        failed_samples = []
        
        print(f"🚀 开始批量提取 {len(sample_names)} 个样本的特征...")
        
        for idx, sample_name in enumerate(sample_names, 1):
            try:
                # 提取特征
                feature_vector = self.extract_single_feature(sample_name)
                
                # 创建特征字典
                feature_dict = {"sample_name": sample_name}
                for i, val in enumerate(feature_vector):
                    feature_dict[f"feature_{i}"] = float(val)
                
                features_data.append(feature_dict)
                
                # 显示进度
                if idx % 10 == 0 or idx == len(sample_names):
                    print(f"  [{idx}/{len(sample_names)}] {sample_name}: 特征维度={len(feature_vector)}")
                    
            except Exception as e:
                print(f"  ❌ {sample_name}: 提取失败 - {str(e)[:100]}...")
                failed_samples.append((sample_name, str(e)))
        
        # 创建DataFrame
        features_df = pd.DataFrame(features_data)
        
        # 保存到CSV
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            features_df.to_csv(output_csv, index=False)
            print(f"✅ 特征已保存到: {output_csv}")
        
        # 打印总结
        print(f"🎉 批量提取完成!")
        print(f"   成功: {len(features_df)}/{len(sample_names)}")
        if failed_samples:
            print(f"   失败: {len(failed_samples)}")
            for sample, error in failed_samples[:3]:  # 只显示前3个失败样本
                print(f"     - {sample}: {error}")
        
        return features_df
    
    def get_all_sample_names(self) -> List[str]:
        """
        获取预处理文件夹中的所有样本名称
        
        Returns:
            样本名称列表
        """
        if not os.path.exists(self.preprocessed_folder):
            raise FileNotFoundError(f"预处理文件夹不存在: {self.preprocessed_folder}")
        
        all_files = os.listdir(self.preprocessed_folder)
        
        # 提取所有.b2nd文件（不包括_seg.b2nd文件）
        sample_names = []
        for file in all_files:
            if file.endswith('.b2nd') and not file.endswith('_seg.b2nd'):
                sample_name = file.replace('.b2nd', '')
                sample_names.append(sample_name)
        
        # 去重并排序
        sample_names = sorted(list(set(sample_names)))
        
        return sample_names

# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 配置路径
    CONFIG = {
        "preprocessed_folder": "D:\\Graduate\\nnUNet_preprocessed\\Dataset002_ISLE24\\nnUNetPlans_3d_fullres",
        "checkpoint_path": "D:\\Graduate\\Graduate\\model_nnUnet\\dwi_adc_model\\checkpoint_best.pth",
        "plans_path": "D:\\Graduate\\Graduate\\model_nnUnet\\dwi_adc_model\\nnUNetPlans.json",
        "dataset_json_path": "D:\\Graduate\\Graduate\\model_nnUnet\\dwi_adc_model\\dataset.json",
        "output_csv": "D:\\Graduate\\Features\\nnunet_features.csv"
    }
    
    try:
        # 1. 初始化提取器
        extractor = nnUNetFeatureExtractor(
            preprocessed_folder=CONFIG["preprocessed_folder"],
            checkpoint_path=CONFIG["checkpoint_path"],
            plans_path=CONFIG["plans_path"],
            dataset_json_path=CONFIG["dataset_json_path"]
        )
        
        # 2. 获取所有样本名称
        sample_names = extractor.get_all_sample_names()
        print(f"找到 {len(sample_names)} 个样本")
        
        # 3. 批量提取特征（处理前10个样本作为示例）
        subset_samples = sample_names[:10] if len(sample_names) > 10 else sample_names
        features_df = extractor.extract_batch_features(
            sample_names=subset_samples,
            output_csv=CONFIG["output_csv"]
        )
        
        # 4. 显示特征信息
        print(f"\n📊 特征统计:")
        print(f"   特征维度: {len(features_df.columns) - 1}")  # 减去sample_name列
        print(f"   样本数量: {len(features_df)}")
        print(f"   特征列示例: {list(features_df.columns)[1:5]}...")  # 显示前4个特征列
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
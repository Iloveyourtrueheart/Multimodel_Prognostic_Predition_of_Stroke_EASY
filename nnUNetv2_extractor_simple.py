import os
import torch
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import join
import blosc2
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

# ================== 配置参数 ==================
PREPROCESSED_FOLDER = "D:\\Graduate\\nnUNet_preprocessed\\Dataset002_ISLE24\\nnUNetPlans_3d_fullres"
CHECKPOINT_PATH = "D:\\Graduate\\Graduate\\model_nnUnet\\dwi_adc_model\\checkpoint_best.pth"
PLANS_PATH = "D:\\Graduate\\Graduate\\model_nnUnet\\dwi_adc_model\\nnUNetPlans.json"
DATASET_JSON_PATH = "D:\\Graduate\\Graduate\\model_nnUnet\\dwi_adc_model\\dataset.json"
OUTPUT_CSV = "D:\\Graduate\\Features\\nnunet_features_max_and_mean.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES = 149  # 处理样本数

# ================== 特征提取器 ==================
class FeatureExtractor:
    def __init__(self):
        # 加载配置
        plans = pd.read_json(PLANS_PATH, typ='series').to_dict() if PLANS_PATH.endswith('.json') else __import__('json').load(open(PLANS_PATH))
        dataset_json = pd.read_json(DATASET_JSON_PATH, typ='series').to_dict() if DATASET_JSON_PATH.endswith('.json') else __import__('json').load(open(DATASET_JSON_PATH))
        
        plans_manager = PlansManager(plans)
        config_manager = plans_manager.get_configuration("3d_fullres")
        
        # 获取通道数
        self.num_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
        num_output = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
        
        # 重建网络
        self.net = get_network_from_plans(
            config_manager.network_arch_class_name,
            config_manager.network_arch_init_kwargs,
            config_manager.network_arch_init_kwargs_req_import,
            self.num_channels, num_output, allow_init=True, deep_supervision=False
        ).to(DEVICE)
        # print(dir(self.net))  # 查看网络有哪些属性和方法
        # print(type(self.net))  # 查看网络类型
        # 加载权重
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = state_dict.get('network_weights', state_dict)
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)
        self.net.eval()
    
    def extract(self, x):
        """提取特征: x shape [1, C, D, H, W]"""
        with torch.no_grad():
            features = self.net.encoder(x)
            bottleneck = features[-1] if isinstance(features, (list, tuple)) else features
            # 全局平均池化 [1, C, D, H, W] -> [1, C]
            #global_feat = torch.mean(bottleneck, dim=[2, 3, 4]).cpu().numpy().flatten()


            # global_feat = torch.max(bottleneck, dim=[2, 3, 4])[0]  # 全局最大池化
            # 或者同时使用平均和最大
            avg_pool = torch.mean(bottleneck, dim=[2, 3, 4])
            max_pool = torch.amax(bottleneck, dim=[2, 3, 4])
            global_feat = torch.cat([avg_pool, max_pool], dim=1)  # [1, 2C]
            global_feat = global_feat.cpu().numpy().flatten()  # [2C] 一维数组
        return global_feat

# ================== 数据加载 ==================
def load_data(sample_name):
    """加载单个样本的预处理数据"""
    data_path = join(PREPROCESSED_FOLDER, sample_name + '.b2nd')
    data = blosc2.open(urlpath=data_path, mode='r')[:]
    
    # 转换为 [1, C, D, H, W]
    if len(data.shape) == 4:
        data = data[np.newaxis, ...]
    elif len(data.shape) == 5 and data.shape[0] > 1:
        data = data[0:1, ...]
    
    return torch.from_numpy(data.astype(np.float32)).to(DEVICE)

# ================== 获取样本列表 ==================
def get_samples():
    """获取所有.b2nd文件（排除分割文件）"""
    files = os.listdir(PREPROCESSED_FOLDER)
    samples = [f.replace('.b2nd', '') for f in files 
               if f.endswith('.b2nd') and not f.endswith('_seg.b2nd')]
    return sorted(set(samples))[:MAX_SAMPLES]

# ================== 主流程 ==================
def main():
    print("开始特征提取...")
    
    # 初始化
    extractor = FeatureExtractor()
    samples = get_samples()
    print(f"找到 {len(samples)} 个样本")
    
    # 提取特征
    results = []
    for i, name in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {name}")
        try:
            data = load_data(name)
            feat = extractor.extract(data)
            
            # 保存结果
            row = {'sample_name': name}
            row.update({f'f{j}': feat[j] for j in range(len(feat))})
            results.append(row)
        except Exception as e:
            print(f"  失败: {e}")
    
    # 保存CSV
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"完成！已保存 {len(results)} 个样本，特征维度 {len(results[0])-1}")
        print(f"输出文件: {OUTPUT_CSV}")
    else:
        print("没有成功提取任何特征")

if __name__ == "__main__":
    main()
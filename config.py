# config.py
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    # 基础种子（必须）
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CPU + 显卡种子（必须）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 关键：只开必要确定性，不关闭显卡加速
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 打开！恢复性能
    torch.backends.cudnn.enabled = True    # 打开！恢复性能

class Config:
    """训练配置"""
    # 路径配置
    nnunet_features_path = "D:\\Graduate\\Features\\nnunet_features_max_and_mean.csv"  # 医学影像数据特征
    tabpfn_features_path = "D:\\Graduate\\Features\\tabpfn_features_wholedata.csv"  # 医学临床表格数据特征
    labels_path = "D:\\Graduate\\Features\\labels_with_names.csv"
    
    # 输出目录配置
    output_dir = "D:\\Graduate\\TrainingResults"  # 自定义输出文件夹
    model_save_dir = os.path.join(output_dir, "models")  # 模型保存目录
    results_save_dir = os.path.join(output_dir, "results")  # 结果保存目录
    figures_save_dir = os.path.join(output_dir, "figures")  # 图表保存目录
    
    # 模型参数
    clinical_dim = 192     # TabPFN特征维度
    image_dim = 640        # nnUNet特征维度
    hidden_dim = 512       # 交叉注意力隐藏维度#512 0.7778
    num_heads = 4          # 注意力头数
    dropout = 0.32          # Dropout率
    
    # 训练参数
    batch_size = 32
    learning_rate = 5e-3
    num_epochs = 100
    patience = 50          # 早停耐心值（基于训练集）
    
    # 数据划分 - 只分训练集和测试集
    test_size = 0.3
    random_seed = 42 
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        """初始化时创建必要的目录"""
        self._create_directories()
    
    def _create_directories(self):
        """创建输出目录"""
        directories = [self.output_dir, self.model_save_dir, 
                      self.results_save_dir, self.figures_save_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 创建目录: {directory}")
        
        # 设置具体保存路径
        self.model_save_path = os.path.join(self.model_save_dir, "fusion_model.pth")
        self.results_save_path = os.path.join(self.results_save_dir, "training_results.csv")
        self.training_curves_path = os.path.join(self.figures_save_dir, "training_curves.png")
        self.confusion_matrix_path = os.path.join(self.figures_save_dir, "confusion_matrix.png")
        self.predictions_save_path = os.path.join(self.results_save_dir, "test_predictions.csv")
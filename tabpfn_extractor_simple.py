import numpy as np
import pandas as pd
import os
from tabpfn_extensions import TabPFNClassifier, TabPFNEmbedding
from typing import List, Optional

class TabPNFFeatureExtractor:
    """
    TabPFN特征提取器 - 提取192维特征嵌入
    """
    
    def __init__(self, model_path: str, data_path: str, has_header: bool = True):
        """初始化特征提取器"""
        # 设置离线环境
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        self.model_path = model_path
        self.data_path = data_path
        self.feature_dim = 192  # 固定为192维
        
        # 加载数据
        self.X, self.y = self._load_data(data_path, has_header)
        
        # 初始化模型
        self._init_model()
        
        print(f"✅ TabPFN特征提取器初始化完成")
        print(f"   数据形状: {self.X.shape}")
        print(f"   标签形状: {self.y.shape}")
        print(f"   特征维度: {self.feature_dim}")
    
    def _load_data(self, data_path: str, has_header: bool) -> tuple:
        """加载CSV数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        df = pd.read_csv(data_path) if has_header else pd.read_csv(data_path, header=None)
        
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int32)
        
        print(f"✅ 数据加载完成: {X.shape}, {y.shape}")
        return X, y
    
    def _init_model(self):
        """初始化TabPFN模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.classifier = TabPFNClassifier(
            n_estimators=1,
            device="cuda",
            model_path=self.model_path,
            ignore_pretraining_limits=False
        )
        
        print("正在拟合模型...")
        self.classifier.fit(self.X, self.y)
        print("✅ 模型拟合完成")
        
        self.embedding_extractor = TabPFNEmbedding(tabpfn_clf=self.classifier, n_fold=10)
    
    def extract_features(self, indices: Optional[List[int]] = None, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        提取192维特征
        
        Args:
            indices: 要提取的样本索引列表，None表示所有样本
            output_csv: 输出CSV文件路径（可选）
            
        Returns:
            特征DataFrame，列: ['sample_index', 'feature_0', ..., 'feature_191']
        """
        if indices is None:
            indices = list(range(len(self.X)))
        
        features = []
        for idx in indices:
            try:
                sample = self.X[idx:idx+1]
                
                embedding = self.embedding_extractor.get_embeddings(
                    X_train=self.X,
                    y_train=self.y,
                    X=sample,
                    data_source="test"
                )  # 形状: (1, 192)
                
                # 展平为192维向量
                feature_vector = embedding[0]  # 直接取第一行，形状: (192,)
                features.append(feature_vector)
                
            except Exception as e:
                print(f"❌ 提取样本 {idx} 失败: {e}")
                raise
        
        # 创建DataFrame
        feature_matrix = np.array(features)  # 形状: (n_samples, 192)
        columns = [f"feature_{i}" for i in range(self.feature_dim)]
        features_df = pd.DataFrame(feature_matrix, columns=columns)
        features_df.insert(0, "sample_index", indices)
        
        # 保存到CSV
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            features_df.to_csv(output_csv, index=False)
            print(f"✅ 特征已保存到: {output_csv}")
        
        print(f"✅ 成功提取 {len(features_df)} 个样本的192维特征")
        return features_df
    
    def get_info(self) -> dict:
        """获取数据集信息"""
        return {
            "num_samples": len(self.X),
            "num_features": self.X.shape[1],
            "num_classes": len(np.unique(self.y)),
            "class_distribution": dict(zip(*np.unique(self.y, return_counts=True))),
            "tabpfn_feature_dim": self.feature_dim
        }

# ====================== 使用示例 ======================
if __name__ == "__main__":
    CONFIG = {
        "model_path": "D:\\Graduate\\Graduate\\model_quanzhong\\tabpfn-v2.5-classifier-v2.5_real.ckpt",
        "data_path": "D:\\Graduate\\xlsx\\wholedata.csv",
        "output_csv": "D:\\Graduate\\Features\\tabpfn_features_wholedata_testbug4.csv"
    }
    
    try:
        # 1. 初始化提取器
        extractor = TabPNFFeatureExtractor(
            model_path=CONFIG["model_path"],
            data_path=CONFIG["data_path"],
            has_header=True
        )
        
        # 2. 查看数据集信息
        info = extractor.get_info()
        print(f"\n📊 数据集信息:")
        for k, v in info.items():
            print(f"   {k}: {v}")
        
        # 3. 提取所有样本的192维特征
        features_df = extractor.extract_features(output_csv=CONFIG["output_csv"])
        
        # 4. 验证结果
        print(f"\n📊 提取结果验证:")
        print(f"   特征矩阵形状: {features_df.shape}")
        print(f"   特征维度: {features_df.shape[1]-1}")
        print(f"   是否正确: {features_df.shape[1]-1 == 192}")
        print(f"   前5个特征列: {list(features_df.columns[1:6])}")
        print(f"   特征值范围: [{features_df.iloc[:, 1:].min().min():.4f}, {features_df.iloc[:, 1:].max().max():.4f}]")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
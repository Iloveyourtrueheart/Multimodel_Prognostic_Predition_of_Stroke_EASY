import numpy as np
import pandas as pd
import os
from tabpfn_extensions import TabPFNClassifier, TabPFNEmbedding
from typing import List, Tuple, Optional

# ====================== TabPFN特征提取器 ======================
class TabPNFFeatureExtractor:
    """
    简化版TabPFN特征提取器
    功能：加载TabPFN模型，提取表格数据的特征嵌入（192维）
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        has_header: bool = True
    ):
        """
        初始化特征提取器
        
        Args:
            model_path: TabPFN模型权重文件路径 (.ckpt)
            data_path: CSV数据文件路径
            has_header: CSV文件是否有表头，默认True
        """
        # 设置离线环境
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # 保存参数
        self.model_path = model_path
        self.data_path = data_path
        self.has_header = has_header
        
        # 加载数据和模型
        self.X, self.y = self._load_data()
        self._init_model()
        
        print(f"✅ TabPFN特征提取器初始化完成")
        print(f"   数据形状: {self.X.shape}")
        print(f"   标签形状: {self.y.shape}")
        print(f"   特征维度: 192 (TabPFN嵌入维度)")
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载CSV数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 读取CSV
        if self.has_header:
            df = pd.read_csv(self.data_path)
        else:
            df = pd.read_csv(self.data_path, header=None)
        
        # 拆分特征和标签（最后一列为标签）
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int32)
        
        print(f"✅ 数据加载完成: {X.shape}, {y.shape}")
        return X, y
    
    def _init_model(self):
        """初始化TabPFN模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 初始化分类器
        self.classifier = TabPFNClassifier(
            n_estimators=1,
            device="auto",
            model_path=self.model_path,
            ignore_pretraining_limits=False
        )
        
        # 拟合模型（对齐数据分布）
        print("正在拟合模型...")
        self.classifier.fit(self.X, self.y)
        print("✅ 模型拟合完成")
        
        # 初始化特征提取器
        self.embedding_extractor = TabPFNEmbedding(
            tabpfn_clf=self.classifier,
            n_fold=0
        )
    
    def extract_single_feature(self, sample_index: int) -> np.ndarray:
        """
        提取单个样本的特征向量
        
        Args:
            sample_index: 样本索引（从0开始）
            
        Returns:
            特征向量 (192维)
        """
        # 验证索引范围
        if not (0 <= sample_index < len(self.X)):
            raise IndexError(f"索引 {sample_index} 超出范围 [0, {len(self.X)-1}]")
        
        # 获取单个样本，保持二维形状
        sample = self.X[sample_index:sample_index+1]  # 形状: (1, n_features)
        
        # 提取特征嵌入
        embedding = self.embedding_extractor.get_embeddings(
            X_train=self.X,
            y_train=self.y,
            X=sample,
            data_source="test"
        )
        feature_vector = embedding.T.flatten()
        
        return feature_vector
    
    def extract_batch_features(self, sample_indices: List[int]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            sample_indices: 样本索引列表
            
        Returns:
            特征矩阵 (n_samples, 192)
        """
        features = []
        
        for idx in sample_indices:
            try:
                feature_vector = self.extract_single_feature(idx)
                features.append(feature_vector)
            except Exception as e:
                print(f"❌ 提取样本 {idx} 失败: {e}")
                # 可以选择跳过失败样本或抛出异常
                raise
        
        return np.array(features)
    
    def extract_all_features(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        提取所有样本的特征
        
        Args:
            output_csv: 输出CSV文件路径（可选）
            
        Returns:
            包含所有特征的DataFrame，列: ['sample_index', 'feature_0', ..., 'feature_191']
        """
        all_indices = list(range(len(self.X)))
        feature_matrix = self.extract_batch_features(all_indices)
        
        # 创建DataFrame
        features_data = []
        for idx, feature_vector in zip(all_indices, feature_matrix):
            feature_dict = {"sample_index": idx}
            for i, val in enumerate(feature_vector):
                feature_dict[f"feature_{i}"] = float(val)
            features_data.append(feature_dict)
        
        features_df = pd.DataFrame(features_data)
        
        # 保存到CSV
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            features_df.to_csv(output_csv, index=False)
            print(f"✅ 特征已保存到: {output_csv}")
        
        print(f"✅ 成功提取 {len(features_df)} 个样本的特征")
        return features_df
    
    def get_sample_info(self, sample_index: int) -> dict:
        """
        获取样本信息
        
        Args:
            sample_index: 样本索引
            
        Returns:
            包含样本信息的字典
        """
        if not (0 <= sample_index < len(self.X)):
            raise IndexError(f"索引 {sample_index} 超出范围 [0, {len(self.X)-1}]")
        
        return {
            "index": sample_index,
            "features": self.X[sample_index],
            "label": self.y[sample_index],
            "num_features": self.X.shape[1]
        }
    
    def get_dataset_info(self) -> dict:
        """
        获取数据集信息
        
        Returns:
            数据集信息字典
        """
        return {
            "num_samples": len(self.X),
            "num_features": self.X.shape[1],
            "num_classes": len(np.unique(self.y)),
            "class_distribution": dict(zip(*np.unique(self.y, return_counts=True)))
        }

# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 配置路径
    CONFIG = {
        "model_path": "D:\\Graduate\\Graduate\\model_quanzhong\\tabpfn-v2.5-classifier-v2.5_real.ckpt",
        "data_path": "D:\\Graduate\\xlsx\\wholedata.csv",
        "output_csv": "D:\\Graduate\\Features\\tabpfn_features_wholedata_testbug1.csv"
    }
    
    try:
        # 1. 初始化提取器
        extractor = TabPNFFeatureExtractor(
            model_path=CONFIG["model_path"],
            data_path=CONFIG["data_path"],
            has_header=True
        )
        
        # 2. 获取数据集信息
        dataset_info = extractor.get_dataset_info()
        print(f"📊 数据集信息:")
        print(f"   样本数量: {dataset_info['num_samples']}")
        print(f"   特征数量: {dataset_info['num_features']}")
        print(f"   类别数量: {dataset_info['num_classes']}")
        print(f"   类别分布: {dataset_info['class_distribution']}")
        
        # 3. 提取单个样本特征（示例）
        sample_idx = 5
        sample_feature = extractor.extract_single_feature(sample_idx)
        sample_info = extractor.get_sample_info(sample_idx)
        print(f"\n📋 样本 {sample_idx} 信息:")
        print(f"   特征向量形状: {sample_feature.shape}")
        print(f"   原始标签: {sample_info['label']}")
        
        # 4. 提取所有样本特征
        features_df = extractor.extract_all_features(
            output_csv=CONFIG["output_csv"]
        )
        
        # 5. 显示特征信息
        print(f"\n📊 特征统计:")
        print(f"   特征维度: {len(features_df.columns) - 1}")  # 减去sample_index列
        print(f"   样本数量: {len(features_df)}")
        print(f"   特征列示例: {list(features_df.columns)[1:5]}...")
        
        # 6. 验证特征提取正确性
        print(f"\n🧪 特征验证:")
        print(f"   特征形状检查: {features_df.shape[1]-1 == 192}")
        print(f"   特征无NaN值: {not features_df.iloc[:, 1:].isna().any().any()}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
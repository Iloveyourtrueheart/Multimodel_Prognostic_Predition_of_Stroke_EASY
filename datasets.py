# datasets.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MultiModalDataset(Dataset):
    """
    多模态数据集
    加载并对齐临床特征和影像特征
    """
    def __init__(self, clinical_features, image_features, labels, transform=None):
        """
        Args:
            clinical_features: 临床特征矩阵 (n_samples, 192)
            image_features: 影像特征矩阵 (n_samples, 320)
            labels: 标签数组 (n_samples,)
        """
        self.clinical_features = torch.FloatTensor(clinical_features)
        self.image_features = torch.FloatTensor(image_features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        # 验证数据一致性
        self._validate_data()
    
    def _validate_data(self):
        """验证数据一致性"""
        assert len(self.clinical_features) == len(self.image_features) == len(self.labels), \
            f"数据长度不一致: 临床({len(self.clinical_features)}), 影像({len(self.image_features)}), 标签({len(self.labels)})"
        
        print(f"✅ 数据集验证通过: {len(self)} 个样本")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        clinical = self.clinical_features[idx]
        image = self.image_features[idx]
        label = self.labels[idx]
        
        if self.transform:
            clinical = self.transform(clinical)
            image = self.transform(image)
        
        return {
            'clinical': clinical,
            'image': image,
            'label': label
        }

# class MultiModalDataset(Dataset):
#     def __init__(self, clinical_features, image_features, labels, mode='train', aug_prob=0.3):
#         self.clinical = torch.FloatTensor(clinical_features)
#         self.image = torch.FloatTensor(image_features)
#         self.labels = torch.LongTensor(labels)
#         self.mode = mode
#         self.aug_prob = aug_prob

#     def _augment_feature(self, feat):
#         # 测试集 不增强
#         if self.mode != 'train':
#             return feat
        
#         # 随机概率增强
#         if random.random() > self.aug_prob:
#             return feat

#         # 1. 轻微高斯噪声
#         noise = torch.randn_like(feat) * 0.01
#         feat = feat + noise

#         # 2. 微小缩放
#         scale = random.uniform(0.97, 1.03)
#         feat = feat * scale

#         # 3. 随机遮挡少量特征（抗过拟合）
#         mask = torch.bernoulli(torch.full_like(feat, 0.98))
#         feat = feat * mask

#         return feat.clamp(-5, 5)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         clinical = self.clinical[idx]
#         image = self.image[idx]
#         label = self.labels[idx]

#         # 训练集自动增强
#         clinical = self._augment_feature(clinical)
#         image = self._augment_feature(image)

#         return {
#             'clinical': clinical,
#             'image': image,
#             'label': label
#         }

def load_and_prepare_data(config):
    """
    加载特征数据和标签，进行预处理和划分
    
    Returns:
        train_loader, test_loader, data_info
    """
    print("📥 加载数据...")
    
    # 1. 加载特征数据
    nnunet_df = pd.read_csv(config.nnunet_features_path)
    tabpfn_df = pd.read_csv(config.tabpfn_features_path)
    
    print(f"   nnUNet特征: {len(nnunet_df)} 个样本")
    print(f"   TabPFN特征: {len(tabpfn_df)} 个样本")
    
    # 2. 对齐样本（按sample_name排序）
    nnunet_df = nnunet_df.sort_values('sample_name').reset_index(drop=True)
    tabpfn_df = tabpfn_df.sort_values('sample_name').reset_index(drop=True)
    
    # 验证样本一致性
    nnunet_samples = nnunet_df['sample_name'].tolist()
    tabpfn_samples = tabpfn_df['sample_name'].tolist()
    
    if nnunet_samples != tabpfn_samples:
        print("⚠️  样本名称不完全一致，尝试匹配...")
        # 找出共同的样本
        common_samples = set(nnunet_samples) & set(tabpfn_samples)
        common_samples = sorted(list(common_samples))
        
        nnunet_df = nnunet_df[nnunet_df['sample_name'].isin(common_samples)].sort_values('sample_name')
        tabpfn_df = tabpfn_df[tabpfn_df['sample_name'].isin(common_samples)].sort_values('sample_name')
        
        nnunet_df = nnunet_df.reset_index(drop=True)
        tabpfn_df = tabpfn_df.reset_index(drop=True)
    
    print(f"✅ 对齐后样本数: {len(nnunet_df)}")
    
    # 3. 提取特征矩阵
    # X_nnunet = nnunet_df.filter(regex='^feature_').values.astype(np.float32)#之前的特征维度是320
    X_nnunet = nnunet_df.filter(regex='^f').values.astype(np.float32)
    X_tabpfn = tabpfn_df.filter(regex='^feature_').values.astype(np.float32)
    
    print(f"   nnUNet特征形状: {X_nnunet.shape}")
    print(f"   TabPFN特征形状: {X_tabpfn.shape}")
    
    # 4. 加载标签
    if not os.path.exists(config.labels_path):
        raise FileNotFoundError(f"标签文件不存在: {config.labels_path}")
    
    labels_df = pd.read_csv(config.labels_path)
    
    # 确保标签与特征对齐
    labels_df = labels_df.sort_values('sample_name').reset_index(drop=True)
    labels_df = labels_df[labels_df['sample_name'].isin(nnunet_df['sample_name'])]
    
    # 提取标签
    y = labels_df['label'].values.astype(np.int64)
    
    # 检查二分类标签
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        print(f"⚠️  标签不是严格的二分类: {unique_labels}")
        # 转换为0/1
        y = (y > np.median(y)).astype(np.int64)
    
    print(f"✅ 标签加载完成: {len(y)} 个标签")
    print(f"   类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 5. 数据标准化
    print("📊 数据标准化...")
    
    scaler_nnunet = StandardScaler()
    scaler_tabpfn = StandardScaler()
    
    X_nnunet = scaler_nnunet.fit_transform(X_nnunet)
    X_tabpfn = scaler_tabpfn.fit_transform(X_tabpfn)
    
    # 6. 数据划分（保持类别分布）- 只分训练集和测试集
    indices = np.arange(len(y))
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y
    )
    
    # 创建数据集
    train_dataset = MultiModalDataset(
        clinical_features=X_tabpfn[train_idx],
        image_features=X_nnunet[train_idx],
        labels=y[train_idx],
        #mode='train',  # 训练集启用增强，
        #aug_prob=0.3  # 30%概率增强每个样本
    )
    
    test_dataset = MultiModalDataset(
        clinical_features=X_tabpfn[test_idx],
        image_features=X_nnunet[test_idx],
        labels=y[test_idx],
        #mode='test'  # 测试集不增强
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows可能需要设为0
        pin_memory=True if config.device.type == 'cuda' else False,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False,
        drop_last=False
    )
    
    print(f"📊 数据划分完成:")
    print(f"   训练集: {len(train_idx)} 个样本")
    print(f"   测试集: {len(test_idx)} 个样本")
    
    # 保存数据信息
    data_info = {
        'scaler_nnunet': scaler_nnunet,
        'scaler_tabpfn': scaler_tabpfn,
        'train_indices': train_idx,
        'test_indices': test_idx,
        'train_labels': y[train_idx],
        'test_labels': y[test_idx],
        'sample_names': nnunet_df['sample_name'].tolist(),
        'all_labels': y
    }
    
    # 保存数据划分信息到文件
    data_info_path = os.path.join(config.results_save_dir, "data_split_info.csv")
    data_split_df = pd.DataFrame({
        'sample_name': [data_info['sample_names'][i] for i in indices],
        'split': ['train' if i in train_idx else 'test' for i in indices],
        'label': y
    })
    data_split_df.to_csv(data_info_path, index=False)
    print(f"💾 数据划分信息保存到: {data_info_path}")
    
    return train_loader, test_loader, data_info
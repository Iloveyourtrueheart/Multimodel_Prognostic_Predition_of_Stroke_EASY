import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 配置参数 ======================
class Config:
    """训练配置"""
    # 路径配置
    nnunet_features_path = "D:\\Graduate\\Features\\nnunet_features_with_names.csv"
    tabpfn_features_path = "D:\\Graduate\\Features\\tabpfn_features_no3mRS.csv"
    labels_path = "D:\\Graduate\\Features\\labels_with_names.csv"
    
    # 输出目录配置
    output_dir = "D:\\Graduate\\TrainingResults"  # 自定义输出文件夹
    model_save_dir = os.path.join(output_dir, "models")  # 模型保存目录
    results_save_dir = os.path.join(output_dir, "results")  # 结果保存目录
    figures_save_dir = os.path.join(output_dir, "figures")  # 图表保存目录
    
    # 模型参数
    clinical_dim = 192     # TabPFN特征维度
    image_dim = 320        # nnUNet特征维度
    hidden_dim = 256       # 交叉注意力隐藏维度
    num_heads = 4          # 注意力头数
    dropout = 0.3          # Dropout率
    
    # 训练参数
    batch_size = 16
    learning_rate = 1e-2
    num_epochs = 500
    patience = 100          # 早停耐心值
    
    # 数据划分
    test_size = 0.2
    val_size = 0.1
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

# ====================== 2. 交叉注意力融合模型 ======================
# ====================== 2. 交叉注意力融合模型 ======================
# ====================== 2. 交叉注意力融合模型 ======================
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 如果输入输出维度不同，需要调整维度
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = F.relu(out)
        out = self.dropout2(out)
        
        return out


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力多模态融合模型
    使用交叉注意力机制融合临床特征和影像特征
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 特征投影层（将不同维度映射到相同维度）
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 2. 交叉注意力层（临床特征作为Query，影像特征作为Key/Value）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 3. 注意力后处理
        self.post_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 4. 带有残差连接的分类头
        # 降维层：256 → 192
        self.initial_fc = nn.Sequential(
            nn.Linear(config.hidden_dim, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(config.dropout * 1.2)
        )
        
        # 两个残差块
        self.residual_block1 = ResidualBlock(192, 128, config.dropout)
        self.residual_block2 = ResidualBlock(128, 96, config.dropout)
        
        # 输出层
        self.final_fc = nn.Linear(96, 2)
        
        # 5. 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_features, image_features):
        """
        前向传播
        
        Args:
            clinical_features: 临床特征 (batch, 192)
            image_features: 影像特征 (batch, 320)
            
        Returns:
            logits: 分类logits (batch, 2)
            attention_weights: 注意力权重 (可选，用于可视化)
        """
        # 1. 特征投影
        clinical_proj = self.clinical_proj(clinical_features)
        image_proj = self.image_proj(image_features)
        
        # 2. 交叉注意力
        clinical_seq = clinical_proj.unsqueeze(1)
        image_seq = image_proj.unsqueeze(1)
        
        attended_features, attention_weights = self.cross_attention(
            query=clinical_seq,
            key=image_seq,
            value=image_seq,
            need_weights=True
        )
        
        attended_features = attended_features.squeeze(1)
        
        # 3. 注意力后处理
        fused_features = self.post_attention(attended_features)
        
        # 4. 带有残差连接的分类头
        x = self.initial_fc(fused_features)      # 256 → 192
        x = self.residual_block1(x)              # 192 → 128 (带残差)
        x = self.residual_block2(x)              # 128 → 96 (带残差)
        logits = self.final_fc(x)                # 96 → 2
        
        return logits, attention_weights

# ====================== 3. 多模态数据集 ======================
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

# ====================== 4. 数据加载与预处理 ======================
def load_and_prepare_data(config):
    """
    加载特征数据和标签，进行预处理和划分
    
    Returns:
        train_loader, val_loader, test_loader, data_info
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
    X_nnunet = nnunet_df.filter(regex='^feature_').values.astype(np.float32)
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
    from sklearn.preprocessing import StandardScaler
    
    scaler_nnunet = StandardScaler()
    scaler_tabpfn = StandardScaler()
    
    X_nnunet = scaler_nnunet.fit_transform(X_nnunet)
    X_tabpfn = scaler_tabpfn.fit_transform(X_tabpfn)
    
    # 6. 数据划分（保持类别分布）
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(y))
    
    # 先划分测试集
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y
    )
    
    # 再从训练集中划分验证集
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.val_size/(1-config.test_size),  # 调整比例
        random_state=config.random_seed,
        stratify=y[train_val_idx]
    )
    
    # 创建数据集
    train_dataset = MultiModalDataset(
        clinical_features=X_tabpfn[train_idx],
        image_features=X_nnunet[train_idx],
        labels=y[train_idx]
    )
    
    val_dataset = MultiModalDataset(
        clinical_features=X_tabpfn[val_idx],
        image_features=X_nnunet[val_idx],
        labels=y[val_idx]
    )
    
    test_dataset = MultiModalDataset(
        clinical_features=X_tabpfn[test_idx],
        image_features=X_nnunet[test_idx],
        labels=y[test_idx]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows可能需要设为0
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    print(f"📊 数据划分完成:")
    print(f"   训练集: {len(train_idx)} 个样本")
    print(f"   验证集: {len(val_idx)} 个样本")
    print(f"   测试集: {len(test_idx)} 个样本")
    
    # 保存数据信息
    data_info = {
        'scaler_nnunet': scaler_nnunet,
        'scaler_tabpfn': scaler_tabpfn,
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'train_labels': y[train_idx],
        'val_labels': y[val_idx],
        'test_labels': y[test_idx],
        'sample_names': nnunet_df['sample_name'].tolist(),
        'all_labels': y
    }
    
    # 保存数据划分信息到文件
    data_info_path = os.path.join(config.results_save_dir, "data_split_info.csv")
    data_split_df = pd.DataFrame({
        'sample_name': [data_info['sample_names'][i] for i in indices],
        'split': ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in indices],
        'label': y
    })
    data_split_df.to_csv(data_info_path, index=False)
    print(f"💾 数据划分信息保存到: {data_info_path}")
    
    return train_loader, val_loader, test_loader, data_info

# ====================== 5. 训练与评估函数 ======================
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        clinical = batch['clinical'].to(device)
        image = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits, _ = model(clinical, image)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * clinical.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 学习率调整
    if scheduler:
        scheduler.step()
    
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, return_attention=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_predictions = []
    all_attention_weights = [] if return_attention else None
    
    with torch.no_grad():
        for batch in dataloader:
            clinical = batch['clinical'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits, attention_weights = model(clinical, image)
            loss = criterion(logits, labels)
            
            # 统计
            total_loss += loss.item() * clinical.size(0)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            
            if return_attention:
                all_attention_weights.append(attention_weights.cpu().numpy())
    
    # 合并结果
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_probs, all_labels, all_predictions, all_attention_weights

# ====================== 6. 训练主循环 ======================
def train_model(config):
    """训练多模态融合模型"""
    print("=" * 60)
    print("🚀 开始训练多模态融合模型")
    print(f"📂 输出目录: {config.output_dir}")
    print("=" * 60)
    
    # 1. 加载数据
    train_loader, val_loader, test_loader, data_info = load_and_prepare_data(config)
    
    # 2. 初始化模型
    print("\n🔧 初始化模型...")
    model = CrossAttentionFusion(config)
    model = model.to(config.device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型初始化完成")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 保存模型架构信息
    model_info_path = os.path.join(config.results_save_dir, "model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"模型名称: CrossAttentionFusion\n")
        f.write(f"总参数量: {total_params:,}\n")
        f.write(f"可训练参数量: {trainable_params:,}\n")
        f.write(f"临床特征维度: {config.clinical_dim}\n")
        f.write(f"影像特征维度: {config.image_dim}\n")
        f.write(f"隐藏维度: {config.hidden_dim}\n")
        f.write(f"注意力头数: {config.num_heads}\n")
        f.write(f"Dropout率: {config.dropout}\n")
        f.write(f"\n模型架构:\n")
        f.write(str(model))
    print(f"💾 模型信息保存到: {model_info_path}")
    
    # 3. 定义损失函数和优化器
    # 处理类别不平衡
    class_counts = np.bincount(data_info['train_labels'])
    if len(class_counts) == 2:
        class_weights = torch.FloatTensor([1.0, class_counts[0]/class_counts[1]]).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"⚠️  检测到类别不平衡，使用加权损失: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4  # L2正则化
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 4. 训练循环
    print("\n🔄 开始训练...")
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(config.num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        # 验证
        val_loss, val_acc, _, _, _, _ = evaluate(
            model, val_loader, criterion, config.device
        )
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存每10个epoch的中间模型
        if (epoch + 1) % 10 == 0:
            intermediate_model_path = os.path.join(
                config.model_save_dir, 
                f"fusion_model_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, intermediate_model_path)
            print(f"💾 保存中间模型: {intermediate_model_path}")
        
        # 早停和保存最佳模型
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config.__dict__,
                'data_info': data_info
            }, config.model_save_path)
            print(f"💾 保存最佳模型到: {config.model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"🛑 早停触发，验证指标 {config.patience} 个epoch未提升")
                break
    
    # 5. 加载最佳模型进行测试
    print("\n🧪 在测试集上评估最佳模型...")
    checkpoint = torch.load(config.model_save_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_probs, test_labels, test_predictions, _ = evaluate(
        model, test_loader, criterion, config.device
    )
    
    print(f"📊 测试集结果:")
    print(f"   测试损失: {test_loss:.4f}")
    print(f"   测试准确率: {test_acc:.4f}")
    
    # 6. 计算分类报告
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n📈 分类报告:")
    report = classification_report(test_labels, test_predictions, target_names=['Class 0', 'Class 1'], output_dict=True)
    print(classification_report(test_labels, test_predictions, target_names=['Class 0', 'Class 1']))
    
    # 计算AUC
    if len(np.unique(test_labels)) == 2:
        auc = roc_auc_score(test_labels, test_probs[:, 1])
        print(f"📊 ROC-AUC: {auc:.4f}")
    
    # 7. 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.results_save_path, index=False)
    print(f"💾 训练历史保存到: {config.results_save_path}")
    
    # 8. 保存详细的测试结果
    test_results_df = pd.DataFrame({
        'sample_index': data_info['test_indices'],
        'sample_name': [data_info['sample_names'][i] for i in data_info['test_indices']],
        'true_label': test_labels,
        'predicted_label': test_predictions,
        'prob_class_0': test_probs[:, 0],
        'prob_class_1': test_probs[:, 1],
        'correct': test_labels == test_predictions
    })
    test_results_df.to_csv(config.predictions_save_path, index=False)
    print(f"💾 测试预测结果保存到: {config.predictions_save_path}")
    
    # 9. 保存评估指标总结
    metrics_summary = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'roc_auc': auc if 'auc' in locals() else None,
        'class_0_precision': report['Class 0']['precision'],
        'class_0_recall': report['Class 0']['recall'],
        'class_0_f1': report['Class 0']['f1-score'],
        'class_1_precision': report['Class 1']['precision'],
        'class_1_recall': report['Class 1']['recall'],
        'class_1_f1': report['Class 1']['f1-score'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'best_val_accuracy': best_val_acc,
        'best_val_loss': best_val_loss,
        'total_epochs_trained': len(history),
        'early_stopping_triggered': patience_counter >= config.patience
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_path = os.path.join(config.results_save_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"💾 评估指标总结保存到: {metrics_path}")
    
    # 10. 可视化训练过程
    try:
        import matplotlib.pyplot as plt
        
        # 创建图表目录
        os.makedirs(config.figures_save_dir, exist_ok=True)
        
        # 损失和准确率曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=10)
        
        # 准确率曲线
        axes[1].plot(history_df['epoch'], history_df['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(history_df['epoch'], history_df['val_acc'], label='Val Acc', linewidth=2)
        axes[1].axhline(y=test_acc, color='r', linestyle='--', label='Test Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(config.training_curves_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练曲线保存到: {config.training_curves_path}")
        
        # 混淆矩阵
        fig2, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(test_labels, test_predictions)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 添加文本标签
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Class 0', 'Class 1'])
        ax.set_yticklabels(['Class 0', 'Class 1'])
        
        plt.tight_layout()
        plt.savefig(config.confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"📊 混淆矩阵保存到: {config.confusion_matrix_path}")
        
        # 显示图表
        plt.show()
        
    except ImportError:
        print("⚠️  Matplotlib未安装，跳过可视化")
    
    return model, history_df, test_acc

# ====================== 7. 主函数 ======================
def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    print("🎯 多模态融合模型训练")
    print(f"   设备: {config.device}")
    print(f"   临床特征维度: {config.clinical_dim}")
    print(f"   影像特征维度: {config.image_dim}")
    print(f"   隐藏维度: {config.hidden_dim}")
    print(f"   注意力头数: {config.num_heads}")
    print(f"   批大小: {config.batch_size}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   总epoch数: {config.num_epochs}")
    print(f"\n📂 输出目录结构:")
    print(f"   - 主目录: {config.output_dir}")
    print(f"   - 模型目录: {config.model_save_dir}")
    print(f"   - 结果目录: {config.results_save_dir}")
    print(f"   - 图表目录: {config.figures_save_dir}")
    
    # 确认开始训练
    response = input("\n⚠️  是否开始训练？(y/n): ")
    if response.lower() != 'y':
        print("❌ 用户取消操作")
        return
    
    # 开始训练
    try:
        model, history, test_acc = train_model(config)
        
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print(f"   最佳测试准确率: {test_acc:.4f}")
        print(f"   输出目录: {config.output_dir}")
        print("=" * 60)
        
        # 显示保存的文件
        print("\n📁 生成的文件:")
        print(f"   1. 最佳模型: {config.model_save_path}")
        print(f"   2. 训练历史: {config.results_save_path}")
        print(f"   3. 测试预测: {config.predictions_save_path}")
        print(f"   4. 训练曲线: {config.training_curves_path}")
        print(f"   5. 混淆矩阵: {config.confusion_matrix_path}")
        
        # 下一步建议
        print("\n💡 下一步建议:")
        print("   1. 检查训练曲线，确保没有过拟合")
        print("   2. 分析混淆矩阵，了解模型错误类型")
        print("   3. 可以尝试调整超参数（学习率、dropout等）")
        print("   4. 考虑使用交叉验证获得更稳定的结果")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

# ====================== 8. 直接运行 ======================
if __name__ == "__main__":
    main()
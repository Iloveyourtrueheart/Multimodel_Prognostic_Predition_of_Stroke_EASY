# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
   

# ======================主模型：双向交叉注意力模型======================
class DualCrossAttentionFusion(nn.Module):
    """
    双路交叉注意力多模态融合模型
    使用双向交叉注意力机制融合临床特征和影像特征
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 特征投影层
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),    
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )        

        # 2. 双路交叉注意力层
        # 第一路：临床特征作为Query，影像特征作为Key/Value
        self.cross_attention_clinical_to_image = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 第二路：影像特征作为Query，临床特征作为Key/Value
        self.cross_attention_image_to_clinical = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 3. 注意力后处理（每路都有自己的后处理）
        self.post_attention_clinical = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.post_attention_image = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 4. 特征融合层
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 4),
            nn.LayerNorm(config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid()
        )
        
        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, 2)
        )
        # 6. 初始化权重
        self._initialize_weights()

        #7. 新增融合后归一化层
        self.post_fusion_norm = nn.LayerNorm(config.hidden_dim)

        #dropout layer
        self.dropout_global = nn.Dropout(config.dropout * 1.2)

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
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
            attention_weights: 注意力权重元组 (attn_weights_1, attn_weights_2)
        """
        # 1. 特征投影
        clinical_proj = self.clinical_proj(clinical_features)  # (batch, hidden_dim)
        image_proj = self.image_proj(image_features)          # (batch, hidden_dim)
        
        # 添加序列维度 (batch, seq_len=1, hidden_dim)
        clinical_seq = clinical_proj.unsqueeze(1)  # (batch, 1, hidden_dim)
        image_seq = image_proj.unsqueeze(1)        # (batch, 1, hidden_dim)
        
        # 2. 第一路交叉注意力：临床特征查询影像特征
        attended_clinical, attn_weights_1 = self.cross_attention_clinical_to_image(
            query=clinical_seq,
            key=image_seq,
            value=image_seq,
            need_weights=True
        )
        attended_clinical = attended_clinical.squeeze(1)  # (batch, hidden_dim)
        
        # 3. 第二路交叉注意力：影像特征查询临床特征
        attended_image, attn_weights_2 = self.cross_attention_image_to_clinical(
            query=image_seq,
            key=clinical_seq,
            value=clinical_seq,
            need_weights=True
        )
        attended_image = attended_image.squeeze(1)  # (batch, hidden_dim)
        
        # 4. 注意力后处理（带残差连接）
        fused_clinical = self.post_attention_clinical(attended_clinical) + attended_clinical
        fused_image = self.post_attention_image(attended_image) + attended_image
        
        # 5. 自适应特征融合
        # 拼接两路特征
        combined = torch.cat([fused_clinical, fused_image], dim=-1)  # (batch, hidden_dim*2)
        

        #=============================#自定义融合门控机制=============================#
        # 计算融合门控权重
        gate = self.fusion_gate(combined)  # (batch, hidden_dim)
        fused_clinical_inter = fused_clinical * (1 + gate)  # 临床特征增强
        fused_image_inter = fused_image * (2 - gate)        # 影像特征互补
        fused_features = fused_clinical_inter + fused_image_inter  # 加性融合
        fused_features = fused_features + torch.mul(fused_clinical, fused_image)  # 乘性交互（捕捉模态协同）

        fused_features = self.dropout_global(fused_features)  # 新增全局Dropout
        fused_features = self.post_fusion_norm(fused_features)  # 新增归一化层
        # 6. 分类
        logits = self.classifier(fused_features)
        
        return logits, (attn_weights_1, attn_weights_2)


# ====================== 消融实验模型 - 简单拼接特征+二层全连接神经网络 ======================
class SimpleMLP(nn.Module):
    """
    基础二层全连接神经网络（消融实验基准模型）
    输入：拼接后的临床+影像特征
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config.clinical_dim + config.image_dim  # 拼接特征维度
        
        self.mlp = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.8),
            # 第二层
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            # 分类层
            nn.Linear(config.hidden_dim, 2)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_features, image_features):
        # 拼接特征
        fused = torch.cat([clinical_features, image_features], dim=-1)
        logits = self.mlp(fused)
        # 保持返回格式统一（兼容原有训练逻辑）
        return logits, None


    
# ====================== 消融实验模型 - 去掉门控融合（仅双向注意力+简单加性融合） ======================
class DualCrossAttentionNoGate(nn.Module):
    """
    消融实验模型：去掉门控融合模块
    仅使用双向交叉注意力 + 简单加性融合（无门控机制）
    用于验证门控融合模块的有效性
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 特征投影层（与完整模型相同）
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),    
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )        

        # 2. 双路交叉注意力层（与完整模型相同）
        self.cross_attention_clinical_to_image = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.cross_attention_image_to_clinical = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 3. 注意力后处理（与完整模型相同）
        self.post_attention_clinical = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.post_attention_image = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 4. 【消融】去掉门控融合，改为简单的拼接+线性层融合
        self.simple_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 5. 分类头（与完整模型相同）
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, 2)
        )
        
        # 6. 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_features, image_features):
        # 1. 特征投影
        clinical_proj = self.clinical_proj(clinical_features)
        image_proj = self.image_proj(image_features)
        
        clinical_seq = clinical_proj.unsqueeze(1)
        image_seq = image_proj.unsqueeze(1)
        
        # 2. 双向交叉注意力
        attended_clinical, attn_weights_1 = self.cross_attention_clinical_to_image(
            query=clinical_seq, key=image_seq, value=image_seq, need_weights=True
        )
        attended_clinical = attended_clinical.squeeze(1)
        
        attended_image, attn_weights_2 = self.cross_attention_image_to_clinical(
            query=image_seq, key=clinical_seq, value=clinical_seq, need_weights=True
        )
        attended_image = attended_image.squeeze(1)
        
        # 3. 注意力后处理（残差）
        fused_clinical = self.post_attention_clinical(attended_clinical) + attended_clinical
        fused_image = self.post_attention_image(attended_image) + attended_image
        
        # 4. 【消融】简单拼接融合（替代门控机制）
        combined = torch.cat([fused_clinical, fused_image], dim=-1)
        fused_features = self.simple_fusion(combined)
        
        # 5. 分类
        logits = self.classifier(fused_features)
        
        return logits, (attn_weights_1, attn_weights_2)


# ====================== 对比实验模型 - 单向交叉注意力（影像→临床） ======================
class CrossAttentionImageToClinical(nn.Module):
    """
    单向交叉注意力模型：影像特征作为Query，临床特征作为Key/Value
    仅使用一个方向的交叉注意力（影像查询临床）
    用于验证双向注意力的必要性
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 特征投影层
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),    
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )        

        # 2. 单向交叉注意力层（影像查询临床）
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
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_features, image_features):
        # 1. 特征投影
        clinical_proj = self.clinical_proj(clinical_features)
        image_proj = self.image_proj(image_features)
        
        clinical_seq = clinical_proj.unsqueeze(1)
        image_seq = image_proj.unsqueeze(1)
        
        # 2. 单向交叉注意力：影像(Query) 查询 临床(Key/Value)
        attended_features, attention_weights = self.cross_attention(
            query=image_seq,      # 影像作为Query
            key=clinical_seq,     # 临床作为Key
            value=clinical_seq,   # 临床作为Value
            need_weights=True
        )
        attended_features = attended_features.squeeze(1)
        
        # 3. 注意力后处理（残差）
        fused_features = self.post_attention(attended_features) + attended_features
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        return logits, attention_weights


# ====================== 对比实验模型 - 单向交叉注意力（临床→影像） ======================
class CrossAttentionClinicalToImage(nn.Module):
    """
    单向交叉注意力模型：临床特征作为Query，影像特征作为Key/Value
    仅使用一个方向的交叉注意力（临床查询影像）
    用于验证双向注意力的必要性
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 特征投影层
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),    
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim*2),
            nn.LayerNorm(config.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(config.dropout*0.8),

            nn.LayerNorm(config.hidden_dim*2),
            nn.Linear(config.hidden_dim*2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )        

        # 2. 单向交叉注意力层（临床查询影像）
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
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim // 2, 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_features, image_features):
        # 1. 特征投影
        clinical_proj = self.clinical_proj(clinical_features)
        image_proj = self.image_proj(image_features)
        
        clinical_seq = clinical_proj.unsqueeze(1)
        image_seq = image_proj.unsqueeze(1)
        
        # 2. 单向交叉注意力：临床(Query) 查询 影像(Key/Value)
        attended_features, attention_weights = self.cross_attention(
            query=clinical_seq,   # 临床作为Query
            key=image_seq,        # 影像作为Key
            value=image_seq,      # 影像作为Value
            need_weights=True
        )
        attended_features = attended_features.squeeze(1)
        
        # 3. 注意力后处理（残差）
        fused_features = self.post_attention(attended_features) + attended_features
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        return logits, attention_weights
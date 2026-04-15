# main.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from config import Config, set_seed
from datasets import load_and_prepare_data
# 导入现有模型
from model import (
    DualCrossAttentionFusion,
    SimpleMLP,
    DualCrossAttentionNoGate,
    CrossAttentionImageToClinical,
    CrossAttentionClinicalToImage
)

# ====================== 补全缺失的模型类 ======================
class ClinicalOnlyMLP(nn.Module):
    """仅临床特征 + MLP"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.classifier = nn.Linear(config.hidden_dim // 2, 2)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, clinical, image):
        x = self.encoder(clinical)
        return self.classifier(x), None

class ImageOnlyMLP(nn.Module):
    """仅影像特征 + MLP"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.classifier = nn.Linear(config.hidden_dim // 2, 2)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, clinical, image):
        x = self.encoder(image)
        return self.classifier(x), None

# 消融变体：移除乘性交互
class DualCrossAttentionFusion_NoMul(DualCrossAttentionFusion):
    def forward(self, clinical_features, image_features):
        clinical_proj = self.clinical_proj(clinical_features)
        image_proj = self.image_proj(image_features)
        clinical_seq = clinical_proj.unsqueeze(1)
        image_seq = image_proj.unsqueeze(1)
        attended_clinical, attn_weights_1 = self.cross_attention_clinical_to_image(
            query=clinical_seq, key=image_seq, value=image_seq, need_weights=True)
        attended_clinical = attended_clinical.squeeze(1)
        attended_image, attn_weights_2 = self.cross_attention_image_to_clinical(
            query=image_seq, key=clinical_seq, value=clinical_seq, need_weights=True)
        attended_image = attended_image.squeeze(1)
        fused_clinical = self.post_attention_clinical(attended_clinical) + attended_clinical
        fused_image = self.post_attention_image(attended_image) + attended_image
        combined = torch.cat([fused_clinical, fused_image], dim=-1)
        gate = self.fusion_gate(combined)
        fused_clinical_inter = fused_clinical * (1 + gate)
        fused_image_inter = fused_image * (2 - gate)
        fused_features = fused_clinical_inter + fused_image_inter   # 无乘性交互
        fused_features = self.dropout_global(fused_features)
        fused_features = self.post_fusion_norm(fused_features)
        logits = self.classifier(fused_features)
        return logits, (attn_weights_1, attn_weights_2)

# 消融变体：移除全局Dropout和后融合归一化
class DualCrossAttentionFusion_NoReg(DualCrossAttentionFusion):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_global = nn.Identity()
        self.post_fusion_norm = nn.Identity()

# 消融变体：简化投影层（单层映射）
class DualCrossAttentionFusion_SimpleProj(DualCrossAttentionFusion):
    def __init__(self, config):
        super().__init__(config)
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.clinical_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

# 消融变体：简化分类头（单层线性）
class DualCrossAttentionFusion_SimpleCls(DualCrossAttentionFusion):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_dim, 2)

# 消融变体：移除残差连接
class DualCrossAttentionFusion_NoResidual(DualCrossAttentionFusion):
    def forward(self, clinical_features, image_features):
        clinical_proj = self.clinical_proj(clinical_features)
        image_proj = self.image_proj(image_features)
        clinical_seq = clinical_proj.unsqueeze(1)
        image_seq = image_proj.unsqueeze(1)
        attended_clinical, attn_weights_1 = self.cross_attention_clinical_to_image(
            query=clinical_seq, key=image_seq, value=image_seq, need_weights=True)
        attended_clinical = attended_clinical.squeeze(1)
        attended_image, attn_weights_2 = self.cross_attention_image_to_clinical(
            query=image_seq, key=clinical_seq, value=clinical_seq, need_weights=True)
        attended_image = attended_image.squeeze(1)
        # 无残差
        fused_clinical = self.post_attention_clinical(attended_clinical)
        fused_image = self.post_attention_image(attended_image)
        combined = torch.cat([fused_clinical, fused_image], dim=-1)
        gate = self.fusion_gate(combined)
        fused_clinical_inter = fused_clinical * (1 + gate)
        fused_image_inter = fused_image * (2 - gate)
        fused_features = fused_clinical_inter + fused_image_inter
        fused_features = fused_features + torch.mul(fused_clinical, fused_image)
        fused_features = self.dropout_global(fused_features)
        fused_features = self.post_fusion_norm(fused_features)
        logits = self.classifier(fused_features)
        return logits, (attn_weights_1, attn_weights_2)

# ====================== 训练函数（支持动态模型） ======================
def train_single_experiment(config_base, model_class, exp_name, extra_config=None):
    """
    训练单个实验，返回指标字典
    """
    # 创建实验专属输出目录
    exp_output_dir = os.path.join(config_base.output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 动态创建配置副本
    class ExpConfig(Config):
        pass
    exp_config = ExpConfig()
    for key, value in config_base.__dict__.items():
        if not key.startswith('_'):
            setattr(exp_config, key, value)
    exp_config.output_dir = exp_output_dir
    exp_config.model_save_dir = os.path.join(exp_output_dir, "models")
    exp_config.results_save_dir = os.path.join(exp_output_dir, "results")
    exp_config.figures_save_dir = os.path.join(exp_output_dir, "figures")
    exp_config._create_directories()
    exp_config.model_save_path = os.path.join(exp_config.model_save_dir, "best_model.pth")
    exp_config.results_save_path = os.path.join(exp_config.results_save_dir, "training_history.csv")
    exp_config.training_curves_path = os.path.join(exp_config.figures_save_dir, "training_curves.png")
    exp_config.confusion_matrix_path = os.path.join(exp_config.figures_save_dir, "confusion_matrix.png")
    exp_config.predictions_save_path = os.path.join(exp_config.results_save_dir, "test_predictions.csv")
    
    use_weighted_loss = True
    if extra_config and 'use_weighted_loss' in extra_config:
        use_weighted_loss = extra_config['use_weighted_loss']
    
    train_loader, test_loader, data_info = load_and_prepare_data(exp_config)
    
    model = model_class(exp_config).to(exp_config.device)
    
    if use_weighted_loss:
        class_counts = np.bincount(data_info['train_labels'])
        if len(class_counts) == 2:
            class_weights = torch.FloatTensor([1.0, class_counts[0]/class_counts[1]]).to(exp_config.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(device=exp_config.device)
    
    best_test_acc = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(exp_config.num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            clinical = batch['clinical'].to(exp_config.device)
            image = batch['image'].to(exp_config.device)
            labels = batch['label'].to(exp_config.device)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if exp_config.device.type=='cuda' else 'cpu'):
                logits, _ = model(clinical, image)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * clinical.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                clinical = batch['clinical'].to(exp_config.device)
                image = batch['image'].to(exp_config.device)
                labels = batch['label'].to(exp_config.device)
                logits, _ = model(clinical, image)
                loss = criterion(logits, labels)
                test_loss += loss.item() * clinical.size(0)
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        test_loss /= total
        test_acc = correct / total
        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        history.append({'epoch':epoch+1, 'train_loss':train_loss, 'train_acc':train_acc,
                        'test_loss':test_loss, 'test_acc':test_acc})
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), exp_config.model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= exp_config.patience:
                break
    
    model.load_state_dict(torch.load(exp_config.model_save_path, map_location=exp_config.device))
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            clinical = batch['clinical'].to(exp_config.device)
            image = batch['image'].to(exp_config.device)
            labels = batch['label'].to(exp_config.device)
            logits, _ = model(clinical, image)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    acc = (all_preds == all_labels).mean()
    if len(np.unique(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = float('nan')
    report = classification_report(all_labels, all_preds, target_names=['Class0','Class1'], output_dict=True, zero_division=0)
    
    metrics = {
        'Experiment': exp_name,
        'Test Accuracy': acc,
        'Test AUC': auc,
        'Precision (Class1)': report['Class1']['precision'],
        'Recall (Class1)': report['Class1']['recall'],
        'F1 (Class1)': report['Class1']['f1-score'],
        'Macro F1': report['macro avg']['f1-score'],
    }
    return metrics

# ====================== 主函数：运行所有实验 ======================
def main():
    set_seed(42)
    base_config = Config()
    
    unique_experiments = [
        # 对比实验
        ("C1_ClinicalOnlyMLP", ClinicalOnlyMLP, None),
        ("C2_ImageOnlyMLP", ImageOnlyMLP, None),
        ("C3_SimpleMLP", SimpleMLP, None),
        ("C4_CrossAttn_ClinToImg", CrossAttentionClinicalToImage, None),
        ("C5_CrossAttn_ImgToClin", CrossAttentionImageToClinical, None),
        ("C6_BiAttn_NoGate", DualCrossAttentionNoGate, None),
        ("C7_FullModel", DualCrossAttentionFusion, None),
        # 消融实验
        ("A3_NoMultiplicative", DualCrossAttentionFusion_NoMul, None),
        ("A4_NoRegularization", DualCrossAttentionFusion_NoReg, None),
        ("A5_SimpleProjection", DualCrossAttentionFusion_SimpleProj, None),
        ("A6_SimpleClassifier", DualCrossAttentionFusion_SimpleCls, None),
        ("A7_NoResidual", DualCrossAttentionFusion_NoResidual, None),
        ("A8_NoWeightedLoss", DualCrossAttentionFusion, {'use_weighted_loss': False}),
    ]
    
    results = []
    for exp_name, model_class, extra in unique_experiments:
        print("\n" + "="*80)
        print(f"🚀 开始实验: {exp_name}")
        print("="*80)
        try:
            metrics = train_single_experiment(base_config, model_class, exp_name, extra)
            results.append(metrics)
            print(f"✅ {exp_name} 完成: Acc={metrics['Test Accuracy']:.4f}, AUC={metrics['Test AUC']:.4f}")
        except Exception as e:
            print(f"❌ {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("📊 所有实验结果汇总")
    print("="*80)
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Experiment')
    print(results_df.to_string(float_format="%.4f"))
    
    results_df.to_csv(os.path.join(base_config.output_dir, "all_experiments_summary.csv"))
    print(f"\n💾 结果已保存至: {os.path.join(base_config.output_dir, 'all_experiments_summary.csv')}")

if __name__ == "__main__":
    main()
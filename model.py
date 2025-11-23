# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from config import Config
from typing import Optional
from math import log
import numpy as np
from torch.nn import Parameter
from KAN import KANLinear



class DynamicBalancedFocalLoss(nn.Module):
    """动态平衡的Focal Loss损失函数
    
    通过动态调整正负样本权重，结合标签平滑和L1正则化，实现更好的分类性能。
    
    参数:
        gamma (float): focal loss的聚焦参数，用于调节难易样本的权重
        alpha (float): 正负样本的平衡因子
        l1_lambda (float): L1正则化系数
        label_smoothing (float): 标签平滑系数
    """
    def __init__(self, gamma=2.0, alpha=0.5, l1_lambda=1e-5, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.label_smoothing = label_smoothing

    def forward(self, y_true, y_pred, model):
        # 动态平衡因子
        pos_weight = (1 - y_true.mean()).detach()
        neg_weight = y_true.mean().detach()
        
        # 标签平滑
        y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Focal Loss计算
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss
        
        # 动态平衡权重
        balance_weight = y_true * pos_weight + (1 - y_true) * neg_weight
        focal_loss = balance_weight * focal_loss
        
        # 正则化
        l1_reg = torch.tensor(0., device=y_pred.device)
        for param in model.parameters():
            l1_reg += torch.norm(param, p=1)
        
        return focal_loss.mean() + self.l1_lambda * l1_reg


class KANClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = KANLinear(1536, 128)
            self.fc2 = KANLinear(128, 32)
            self.fc3 = KANLinear(32, 1)
            self.dropout = nn.Dropout1d(0.1)
        def forward(self, x):
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)      
            x = self.dropout(x)
            x = self.fc3(x)
            return x
        
class ProteinModel(nn.Module):
    """蛋白质序列分类的主模型
    
    整合多个功能模块，实现端到端的蛋白质序列分类。
    
    主要组件:
        - 特征处理模块：处理不同来源的输入特征
        - 特征融合模块：融合多模态特征
        - 动态分类器：自适应分类层
        - 正则化：包含dropout和权重初始化
    """
    def __init__(self):
        super().__init__()
        # 只为激活的特征创建处理模块
        self.linear = nn.Linear(512,256)
        self.h_linear = nn.Linear(1536,256)
        self.s_linear = nn.Linear(256,256)
        self.st_linear = nn.Linear(256,256)
        self.s_g_linear = nn.Linear(256,256)
        self.s_l_linear = nn.Linear(256,256)
        self.st_g_linear = nn.Linear(256,256)
        self.st_l_linear = nn.Linear(256,256)
        self.drop_path = DropPath(Config.MODEL_CONFIG['drop_path_rate']) if Config.MODEL_CONFIG['use_drop_path'] else nn.Identity()
        self._init_weights()
        self.classifier = KANClassifier()
        self.s_norm = nn.LayerNorm(256)
        self.st_norm = nn.LayerNorm(256)
        # 1. 模态内编码器 (为每种输入特征配备一个小型MLP)
        # 序列模态编码器
        self.seq_global_encoder = nn.Sequential(
            WeightStandardizedConv1d(2176, 256, kernel_size=3, padding=1),
            nn.LayerNorm(256),
            Encoder(num_layers=2, d_model=256, num_heads=8, dff=256*2, rate = 0.1),
            nn.Linear(256, 256)
    )
        self.seq_local_encoder = self.seq_local_encoder = nn.Sequential(
            WeightStandardizedConv1d(2176, 256, kernel_size=3, padding=1),
            nn.LayerNorm(256),
            Encoder(num_layers=2, d_model=256, num_heads=8, dff=256*2, rate = 0.1),
            nn.Linear(256, 256)
    )
        
        # 结构模态编码器
        self.struct_global_encoder = nn.Sequential(
            WeightStandardizedConv1d(1152, 256, kernel_size=3, padding=1),
            nn.LayerNorm(256),
            Encoder(num_layers=2, d_model=256, num_heads=8, dff=256*2, rate = 0.1),
            nn.Linear(256, 256)
    )
        self.struct_local_encoder = nn.Sequential(
            WeightStandardizedConv1d(1152, 256, kernel_size=3, padding=1),
            nn.LayerNorm(256),
            Encoder(num_layers=2, d_model=256, num_heads=8, dff=256*2, rate = 0.1),
            nn.Linear(256, 256)
    )
        
        # 2. 差分特征强化器 (Delta Feature Enhancer)
        self.delta_global_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.delta_local_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.contrastive_projection_head = nn.ModuleDict({
            'seq': nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256) # 投影到对比空间
            ),
            'struct': nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
        })


        self.softmax = nn.Softmax(dim=-1)
        self.tau_cl = 1 
        
        # 第①层（编码前）的投影头：把 raw split 特征投到 256 维用于对比
        self.proj_l1 = nn.ModuleDict({
            'seq': nn.Linear(2176, 256),
            'struct': nn.Linear(1152, 256),
        })

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # 使用较小的标准差进行初始化
                    nn.init.xavier_normal_(param, gain=0.5)
                else:
                    # 对于1D参数，使用较小的均匀分布
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    def _nt_xent(self, z1, z2, tau=1.0):
        """
        z1, z2: [B, D]，同一样本的两种视图（这里对应 WT vs MUT）
        返回：标量 InfoNCE 损失
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.size(0)
        # 2B x D
        z = torch.cat([z1, z2], dim=0)
        # 余弦相似度矩阵 2B x 2B
        sim = torch.mm(z, z.t()) / tau
        # 屏蔽对角
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # 正样本索引：前B的正样本在位置(i, i+B)，后B在位置(i, i-B)
        pos = torch.cat([torch.arange(B, 2*B, device=z.device),
                        torch.arange(0, B, device=z.device)], dim=0)

        # InfoNCE：-log( exp(sim_pos)/sum exp(sim_all) )
        sim_pos = sim[torch.arange(2*B, device=z.device), pos]
        loss = - sim_pos + torch.logsumexp(sim, dim=-1)
        return loss.mean()
    def forward(self, inputs):
        local_pdb = inputs[0]
        global_pdb = inputs[1]

        protT5_local_seq = inputs[2]
        protT5_global_seq = inputs[3]
        esmc_local_seq = inputs[4]
        esmc_global_seq = inputs[5]
        local_seq = torch.cat([protT5_local_seq,esmc_local_seq],dim=-1)
        global_seq = torch.cat([protT5_global_seq,esmc_global_seq],dim=-1)
        local_pdb_wt, local_pdb_vt = self.split_features_by_type(local_pdb)
        local_seq_wt, local_seq_vt = self.split_features_by_type(local_seq)
        global_pdb_wt, global_pdb_vt = self.split_features_by_type(global_pdb)
        global_seq_wt, global_seq_vt = self.split_features_by_type(global_seq)

        # 全局
        z1_s_g = self.proj_l1['seq'](global_seq_wt)   # (B,256)
        z2_s_g = self.proj_l1['seq'](global_seq_vt)
        l1_s_g = self._nt_xent(z1_s_g, z2_s_g, tau=self.tau_cl)

        # 局部
        z1_s_l = self.proj_l1['seq'](local_seq_wt)
        z2_s_l = self.proj_l1['seq'](local_seq_vt)
        l1_s_l = self._nt_xent(z1_s_l, z2_s_l, tau=self.tau_cl)

        # 结构-全局
        z1_st_g = self.proj_l1['struct'](global_pdb_wt)
        z2_st_g = self.proj_l1['struct'](global_pdb_vt)
        l1_st_g = self._nt_xent(z1_st_g, z2_st_g, tau=self.tau_cl)

        # 结构-局部
        z1_st_l = self.proj_l1['struct'](local_pdb_wt)
        z2_st_l = self.proj_l1['struct'](local_pdb_vt)
        l1_st_l = self._nt_xent(z1_st_l, z2_st_l, tau=self.tau_cl)

        L_CL_l1 = l1_s_g + l1_s_l + l1_st_g + l1_st_l

        s_g_wt_enc = self.seq_global_encoder(global_seq_wt.unsqueeze(1)).squeeze(1)
        s_g_mut_enc = self.seq_global_encoder(global_seq_vt.unsqueeze(1)).squeeze(1)
        s_l_wt_enc = self.seq_local_encoder(local_seq_wt.unsqueeze(1)).squeeze(1)
        s_l_mut_enc = self.seq_local_encoder(local_seq_vt.unsqueeze(1)).squeeze(1)
        st_g_wt_enc = self.struct_global_encoder(global_pdb_wt.unsqueeze(1)).squeeze(1)
        st_g_mut_enc = self.struct_global_encoder(global_pdb_vt.unsqueeze(1)).squeeze(1)
        st_l_wt_enc = self.struct_local_encoder(local_pdb_wt.unsqueeze(1)).squeeze(1)
        st_l_mut_enc = self.struct_local_encoder(local_pdb_vt.unsqueeze(1)).squeeze(1)

        p_s_g_wt  = self.contrastive_projection_head['seq'](s_g_wt_enc)
        p_s_g_mut = self.contrastive_projection_head['seq'](s_g_mut_enc)
        l2_s_g = self._nt_xent(p_s_g_wt, p_s_g_mut, tau=self.tau_cl)

        p_s_l_wt  = self.contrastive_projection_head['seq'](s_l_wt_enc)
        p_s_l_mut = self.contrastive_projection_head['seq'](s_l_mut_enc)
        l2_s_l = self._nt_xent(p_s_l_wt, p_s_l_mut, tau=self.tau_cl)

        p_st_g_wt  = self.contrastive_projection_head['struct'](st_g_wt_enc)
        p_st_g_mut = self.contrastive_projection_head['struct'](st_g_mut_enc)
        l2_st_g = self._nt_xent(p_st_g_wt, p_st_g_mut, tau=self.tau_cl)

        p_st_l_wt  = self.contrastive_projection_head['struct'](st_l_wt_enc)
        p_st_l_mut = self.contrastive_projection_head['struct'](st_l_mut_enc)
        l2_st_l = self._nt_xent(p_st_l_wt, p_st_l_mut, tau=self.tau_cl)

        L_CL_l2 = l2_s_g + l2_s_l + l2_st_g + l2_st_l

        L_DCL = L_CL_l1 + L_CL_l2
        delta_s_global = self.delta_global_mlp(s_g_mut_enc - s_g_wt_enc)
        delta_s_local = self.delta_local_mlp(s_l_mut_enc - s_l_wt_enc)
        delta_st_global = self.delta_global_mlp(st_g_mut_enc - st_g_wt_enc)
        delta_st_local = self.delta_local_mlp(st_l_mut_enc - st_l_wt_enc)


        context_s_global = s_g_wt_enc
        context_st_global = st_g_wt_enc

        feature = torch.cat([
            delta_s_global, delta_s_local,
            delta_st_global, delta_st_local, 
            context_s_global, context_st_global
        ], dim=-1)

        tau_sd = 0.1  # 温度参数
        q_high_level = self.softmax(self.s_linear(self.h_linear(feature)) / tau_sd)

        q_s_global = self.softmax(self.s_linear(delta_s_global) / tau_sd)
        q_s_local = self.softmax(self.s_linear(delta_s_local) / tau_sd)
        q_st_global = self.softmax(self.s_linear(delta_st_global) / tau_sd)
        q_st_local = self.softmax(self.s_linear(delta_st_local) / tau_sd)

        # 计算L2损失
        L_s_global_distillation = F.mse_loss(q_s_global, q_high_level)
        L_s_local_distillation = F.mse_loss(q_s_local, q_high_level)
        L_st_global_distillation = F.mse_loss(q_st_global, q_high_level)
        L_st_local_distillation = F.mse_loss(q_st_local, q_high_level)

        # 蒸馏损失总和
        L_distillation = L_s_global_distillation + L_s_local_distillation + L_st_global_distillation + L_st_local_distillation
        return torch.sigmoid(self.classifier(feature).squeeze(-1)),L_distillation,L_DCL

    def split_features_by_type(self,features):
        """
        将特征张量按特征类型拆分
        
        参数:
        features (torch.Tensor): 输入特征张量，形状为 (batch_size, feature_nums, dim)
        type_indices (list): 每种类型的特征索引列表。默认为将特征均分为两部分
        
        返回:
        tuple: 每种类型的特征张量
        """
        feature_wt, feature_vt = features[:, 0, :], features[:, 1, :]

        return feature_wt, feature_vt
    
class ExpandDim(nn.Module):
    """维度扩展模块：在指定位置添加新的维度"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)  # Expand on specified dimension


class SqueezeDim(nn.Module):
    """维度压缩模块：移除指定位置的维度"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)  # Compress specified dimension
    
class PrintShape(nn.Module):
    """形状打印模块：用于调试时查看张量形状"""
    def __init__(self, message):
        super().__init__()
        self.message = message

    def forward(self, x):
        print(f"{self.message}: {x.shape}")
        return x
    
class Transpose(nn.Module):
    """维度转置模块：交换张量的指定维度"""
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
       return x.transpose(self.dim1, self.dim2)
    
class DropPath(nn.Module):
    """随机深度正则化：在训练时随机丢弃部分路径"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            keep_prob = 1 - self.drop_prob
            mask = torch.rand(x.shape[0], 1, device=x.device) < keep_prob
            return x * mask / keep_prob
        return x

class WeightStandardizedConv1d(nn.Conv1d):
    """权重标准化的一维卷积：通过标准化权重提高模型稳定性"""
    def forward(self, x):
        x = x.transpose(1,2)  
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = self.weight.mean(dim=[1,2], keepdim=True)
        var = torch.var(self.weight, dim=[1,2], keepdim=True, unbiased=False)
        normalized_weight = (self.weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups).transpose(1,2)  
class ChannelAttention(nn.Module):
    """通道注意力机制：自适应调整不同通道的重要性
    
    结合平均池化和最大池化，通过全连接层学习通道权重。
    """
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.GELU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (64, 1, 256) -> (64, 256, 1)
        # print(f"[Debug] Transposed input shape: {x.shape}")  # Debug output
        
        # Average pooling and max pooling
        avg_out = self.avg_pool(x).squeeze(-1)  # (batch_size, channels, 1) -> (batch_size, channels)
        max_out = self.max_pool(x).squeeze(-1)  # (batch_size, channels, 1) -> (batch_size, channels)
        # print(f"[Debug] Avg pool output shape: {avg_out.shape}")  # Debug output
        # print(f"[Debug] Max pool output shape: {max_out.shape}")  # Debug output
        
        # Fully connected layer
        avg_out = self.fc(avg_out)  # (batch_size, channels)
        max_out = self.fc(max_out)  # (batch_size, channels)
        out = avg_out + max_out
        # print(f"[Debug] ChannelAttention output shape: {out.unsqueeze(-1).shape}")  # Debug output
        return x * out.unsqueeze(-1)
# 添加新的平均池化类
class Nonlinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_func1 = nn.Linear(input_dim, 768)
        self.activate_func = nn.GELU()
        self.norm_func = nn.LayerNorm(768)
        self.linear_func2 = nn.Linear(768, 256)  # 可继续堆叠更多层
    def forward(self, x):
        # 方案2：带激活函数的MLP        
        x = self.linear_func1(x)
        x = self.activate_func(x)
        x = self.norm_func(x)
        x = self.linear_func2(x)
        return x
class MeanPooling(nn.Module):
    """简单的平均池化层"""
    def __init__(self, hidden_dim):
        super().__init__()
    
    def forward(self, x):
        return x.mean(dim=1)
class MaxPooling(nn.Module):
    """简单的最大池化层"""
    def __init__(self, dim=1):
        """
        初始化最大池化层
        
        参数:
            dim (int): 沿哪个维度进行最大池化，默认为 1
        """
        super(MaxPooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        对输入张量 x 在指定维度进行最大池化
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, feature_dim)
        
        返回:
            torch.Tensor: 最大池化后的张量，形状为 (batch_size, feature_dim)
        """
        return x.max(dim=self.dim)[0]
# 添加新的ResidualBlock类
class ResidualBlock(nn.Module):
    """残差块：实现跳跃连接
    
    参数:
        main_path: 主路径的网络层序列
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, main_path, in_channels, out_channels):
        super().__init__()
        self.main_path = main_path
        # 如果输入输出维度不同，添加1x1投影层
        self.shortcut = (nn.Linear(in_channels, out_channels) 
                        if in_channels != out_channels 
                        else nn.Identity())
        
    def forward(self, x):
        identity = self.shortcut(x)
        return self.main_path(x) + identity

class FeatureRecalibration(nn.Module):
    """特征重校准模块：增强重要特征，抑制冗余特征"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 全局信息
        # x shape: [batch_size, dim]
        scale = x  # 直接使用输入，因为已经是池化后的结果
        scale = self.fc1(scale)  # [batch_size, dim//4]
        scale = F.gelu(scale)
        scale = self.fc2(scale)  # [batch_size, dim]
        scale = torch.sigmoid(scale)
        
        return self.norm(x * scale)


def get_model():
    """模型工厂函数
    
    创建并初始化完整的模型、优化器和学习率调度器。
    
    返回:
        model: 初始化好的模型
        optimizer: AdamW优化器
        scheduler: 序列化的学习率调度器(包含预热和余弦退火)
    """
    #print("######")
    model = ProteinModel().to(Config.DEVICE)
    #print("######")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.OPTIMIZER['lr'],
        weight_decay=Config.OPTIMIZER['weight_decay'],
        betas=Config.OPTIMIZER['betas'],
        eps=Config.OPTIMIZER['eps']
    )
    
    # Create Sequential LR scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            # Linear warmup scheduler
            torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                **Config.SCHEDULER['warmup']
            ),
            # Cosine annealing scheduler
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                **Config.SCHEDULER['cosine']
            )
        ],
        milestones=Config.SCHEDULER['milestones']
    )
    
    return model, optimizer, scheduler
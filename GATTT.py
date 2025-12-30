import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_adj_matrix(data, threshold=0.1, top_k=None):
    """
    计算基于皮尔逊相关系数的邻接矩阵
    data: (Batch, Time, Nodes)
    """
    # 1. 计算所有被试的平均相关矩阵
    # 转换维度计算相关性: (Nodes, Batch * Time)
    flattened_data = data.transpose(0, 2, 1).reshape(200, -1)
    corr_matrix = np.corrcoef(flattened_data)
    
    # 2. 阈值处理
    adj = np.abs(corr_matrix) # 取绝对值，关注连接强度
    if top_k:
        # 每行只保留前 k 个最强连接
        for i in range(len(adj)):
            threshold_val = np.sort(adj[i])[-top_k]
            adj[i][adj[i] < threshold_val] = 0
    else:
        adj[adj < threshold] = 0
        
    # 3. 归一化 (D^-0.5 * A * D^-0.5)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    return torch.FloatTensor(adj_normalized)

# ==========================================
# 新增: Tangent Space Projection (几何流)
# ==========================================
def tangent_space_projection(x):
    """
    在 GPU 上计算切空间投影 (Log-Euclidean Metric)
    Input: x (Batch, Time, Nodes)
    Output: Tangent vector (Batch, Features)
    """
    B, T, N = x.shape
    # 1. 计算协方差矩阵 (Batch, N, N)
    # 中心化
    mean = x.mean(dim=1, keepdim=True)
    x_centered = x - mean
    cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (T - 1)
    
    # 2. 正则化以确保正定 (SPD)，避免 log(0) 或负特征值
    # 添加微小扰动到对角线
    eye = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
    cov = cov + 1e-4 * eye 
    
    # 3. 矩阵对数运算 logm(A) = U * log(S) * U^T
    # 使用 eigh (针对对称矩阵优化)
    L, U = torch.linalg.eigh(cov) 
    
    # 特征值取对数 (Log-Euclidean)
    L = torch.log(torch.clamp(L, min=1e-6))
    L_diag = torch.diag_embed(L)
    
    # 重构矩阵: log(Cov)
    log_cov = torch.bmm(torch.bmm(U, L_diag), U.transpose(1, 2))
    
    # 4. 提取上三角矩阵并拉直 (因为是对称的，下三角冗余)
    # 这一步将 (N, N) 矩阵转化为向量，维度为 N*(N+1)/2
    idx = torch.triu_indices(N, N, offset=0, device=x.device)
    tangent_vec = log_cov[:, idx[0], idx[1]] # (Batch, N*(N+1)/2)
    
    return tangent_vec

# ==========================================
# 新增: GAT Layer (图注意力层)
# ==========================================
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 可学习的线性变换 W
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 可学习的注意力向量 a
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (Batch, Nodes, In_Features)
        # adj: (Nodes, Nodes) 或 (Batch, Nodes, Nodes)
        
        Wh = torch.matmul(h, self.W) # (Batch, N, Out)
        B, N, _ = Wh.size()
        
        # 准备注意力机制的输入: (Batch, N, N, 2*Out)
        # 广播机制：将每个节点特征与其他所有节点特征拼接
        # Wh_rep1: [N, 1, Out] -> [N, N, Out]
        # Wh_rep2: [1, N, Out] -> [N, N, Out]
        
        # 内存优化写法 (避免构建巨大的中间张量):
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        #      = LeakyReLU( (a[:out]^T Wh_i) + (a[out:]^T Wh_j) )
        
        a1 = self.a[:self.out_features, :] # (Out, 1)
        a2 = self.a[self.out_features:, :] # (Out, 1)
        
        # (Batch, N, 1)
        e1 = torch.matmul(Wh, a1) 
        e2 = torch.matmul(Wh, a2)
        
        # 广播相加得到 (Batch, N, N)
        e = self.leakyrelu(e1 + e2.transpose(1, 2))
        
        # 掩码处理 (Connectivity Mask)
        # 只有 adj > 0 的地方才计算注意力，其他地方设为负无穷
        if adj.dim() == 2:
            adj = adj.unsqueeze(0) # (1, N, N)
            
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 聚合特征
        h_prime = torch.matmul(attention, Wh) # (Batch, N, Out)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiScaleSTGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleSTGATBlock, self).__init__()
        
        # 1. GAT 部分 (替代原有的 Linear GCN)
        self.gat = GATLayer(in_channels, out_channels)
        
        # 2. 多尺度 TCN 部分 (保持不变)
        # 分支 1: Kernel 3
        self.tcn3 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=((3-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        # 分支 2: Kernel 5
        self.tcn5 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=((5-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        # 分支 3: Kernel 7
        self.tcn7 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=((7-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        
        # 融合层
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x, adj):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        
        # --- GAT ---
        # GAT 需要 (Batch, Nodes, Features)
        # 这里我们将时间维 T 融合进 Batch 维，对每个时间步的图进行卷积
        x_gat_in = x.permute(0, 2, 3, 1).contiguous().view(B*T, V, C) # (B*T, V, C)
        
        # 如果 adj 是 (V, V)，这里会自动广播
        x_gat_out = self.gat(x_gat_in, adj) # (B*T, V, Out)
        
        # 还原维度: (B, T, V, Out) -> (B, Out, T, V)
        x = x_gat_out.view(B, T, V, -1).permute(0, 3, 1, 2)
        
        # --- Multi-Scale TCN ---
        b3 = self.tcn3(x)
        b5 = self.tcn5(x)
        b7 = self.tcn7(x)
        
        out = torch.cat([b3, b5, b7], dim=1) # (B, 3C, T, V)
        out = self.fusion(out)
        
        return out

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class ASDClassifier(nn.Module):
    def __init__(self, adj, num_nodes=200, time_steps=80, hidden_dim=64, num_sites=17):
        super(ASDClassifier, self).__init__()
        
        self.adj = nn.Parameter(adj.clone(), requires_grad=True)
        self.num_nodes = num_nodes
        
        # ==========================
        # Stream 1: ST-GAT (时空流)
        # ==========================
        self.st_gat1 = MultiScaleSTGATBlock(1, 32)
        self.st_gat2 = MultiScaleSTGATBlock(32, hidden_dim)
        
        self.feature_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5)
        )
        
        self.spatial_proj = nn.Linear(num_nodes * 16, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # ==========================
        # Stream 2: Tangent Space (几何流)
        # ==========================
        # 维度计算: 上三角矩阵元素个数 = N * (N+1) / 2
        tangent_input_dim = num_nodes * (num_nodes + 1) // 2
        
        self.tangent_net = nn.Sequential(
            nn.Linear(tangent_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, hidden_dim), # 投影到与 ST-GAT 相同的维度
            nn.ReLU()
        )
        
        # ==========================
        # Classifiers (Fusion)
        # ==========================
        # 输入维度翻倍 (ST-GAT特征 + Tangent特征)
        fusion_dim = hidden_dim * 2
        
        # 任务分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
        
        # 站点判别器
        self.site_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_sites)
        )

    def forward(self, x, alpha=1.0):
        # x: (Batch, Time, Nodes)
        
        # --- Stream 1: Spatiotemporal (GAT + Transformer) ---
        x_st = x.unsqueeze(1) # (B, 1, T, N) for Conv2d
        
        x_st = self.st_gat1(x_st, self.adj)
        x_st = self.st_gat2(x_st, self.adj) # (B, hidden, T, N)
        
        x_st = x_st.permute(0, 2, 3, 1) # (B, T, N, hidden)
        x_st = self.feature_bottleneck(x_st) # (B, T, N, 16)
        
        B, T, N, _ = x_st.shape
        x_st = x_st.reshape(B, T, -1) # (B, T, N*16)
        
        x_st = self.spatial_proj(x_st)
        x_st = self.transformer(x_st)
        
        feat_st = torch.mean(x_st, dim=1) # (B, hidden_dim) - Temporal Pooling
        
        # --- Stream 2: Tangent Space Geometry ---
        # 这是一个无需训练参数的数学变换，但特征很大，所以在 forward 里算
        feat_geo_raw = tangent_space_projection(x) # (B, 20100)
        feat_geo = self.tangent_net(feat_geo_raw) # (B, hidden_dim)
        
        # --- Feature Fusion ---
        features = torch.cat([feat_st, feat_geo], dim=1) # (B, 2*hidden_dim)
        
        # --- Outputs ---
        # 1. ASD Classification
        class_logits = self.classifier(features)
        
        # 2. Site Discrimination (Gradient Reversal)
        reversed_features = grad_reverse(features, alpha)
        site_logits = self.site_classifier(reversed_features)
        
        return class_logits, site_logits
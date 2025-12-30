import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# 从模型文件导入
from GATTT import ASDClassifier, compute_adj_matrix

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪，保证 Transformer 训练稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
        if (i + 1) % 10 == 0:
            print(f"    Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")
        
    return running_loss / len(loader), accuracy_score(all_labels, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return running_loss / len(loader), acc, f1, all_preds, all_labels

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

def validate_subject_wise(model, X, y, groups, criterion, device, batch_size=32):
    """
    按被试维度进行评估：聚合一个被试所有窗口的预测结果。
    """
    model.eval()
    unique_groups = np.unique(groups)
    all_subject_preds = []
    all_subject_labels = []
    total_loss = 0
    
    # 为了方便索引，这里使用简单的 DataLoader 遍历所有窗口，然后按 group 聚合
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_window_probs = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            # DANN 模式下 model 返回 (class_logits, site_logits, logits_st, logits_geo)
            outputs, _, _, _ = model(batch_x)
            probs = F.softmax(outputs, dim=1)
            all_window_probs.extend(probs.cpu().numpy())
    
    all_window_probs = np.array(all_window_probs)
    
    # 按 Subject ID 聚合
    for sub_id in unique_groups:
        idx = np.where(groups == sub_id)[0]
        # Soft Voting: 平均所有窗口的概率
        avg_prob = np.mean(all_window_probs[idx], axis=0)
        pred = np.argmax(avg_prob)
        
        all_subject_preds.append(pred)
        # 获取该被试的真实标签（所有窗口标签一致）
        all_subject_labels.append(y[idx[0]])
        
    acc = accuracy_score(all_subject_labels, all_subject_preds)
    f1 = f1_score(all_subject_labels, all_subject_preds)
    
    return acc, f1, all_subject_preds, all_subject_labels

def main():
    # 1. 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16 # Reduced from 64 to avoid OOM with Multi-Head GAT
    ACCUM_STEPS = 4 # Simulate Batch Size 64
    LR = 5e-5 
    EPOCHS = 100
    
    # 2. 加载数据
    print("加载增广数据...")
    X = np.load('X_augmented.npy')      
    y = np.load('y_augmented.npy')      
    groups = np.load('groups_augmented.npy') 
    sites = np.load('sites_augmented.npy') # 加载站点标签
    
    num_sites = len(np.unique(sites))
    print(f"检测到站点数量: {num_sites}")
    
    # 3. 单次切分验证 (GroupShuffleSplit)
    from sklearn.model_selection import GroupShuffleSplit
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    print(f"\n启动单次切分验证 (站点对抗版)...")
    print(f"训练集窗口数: {len(train_idx)} | 验证集窗口数: {len(val_idx)}")

    X_train, y_train, sites_train = X[train_idx], y[train_idx], sites[train_idx]
    X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
    
    # 计算邻接矩阵
    adj = compute_adj_matrix(X_train, top_k=15)
    
    # Update DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train), torch.LongTensor(sites_train)), 
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    
    # 初始化模型
    model = ASDClassifier(adj=adj, num_nodes=200, time_steps=80, num_sites=num_sites).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    site_criterion = nn.CrossEntropyLoss()
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    # 使用 Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad() # Initialize gradients
        
        all_preds = []
        all_labels = []
        
        alpha = 1.0
        
        for i, (batch_x, batch_y, batch_sites) in enumerate(train_loader):
            batch_x, batch_y, batch_sites = batch_x.to(device), batch_y.to(device), batch_sites.to(device)
            
            # Dynamic Alpha for DANN
            p = float(i + epoch * len(train_loader)) / EPOCHS / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            class_logits, site_logits, logits_st, logits_geo = model(batch_x, alpha=alpha)
            
            loss_class = class_criterion(class_logits, batch_y)
            loss_st = class_criterion(logits_st, batch_y)
            loss_geo = class_criterion(logits_geo, batch_y)
            loss_site = site_criterion(site_logits, batch_sites)
            
            # Deep Supervision weights: 0.3 for aux branches (reduced to focus on fusion)
            loss = (loss_class + 0.3 * loss_st + 0.3 * loss_geo + 0.05 * loss_site) / ACCUM_STEPS
            
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Cosine Scheduler step (approximately per effective batch)
                scheduler.step(epoch + i / len(train_loader))
            
            running_loss += loss.item() * ACCUM_STEPS # Scale back up for display
            _, preds = torch.max(class_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            if (i + 1) % 20 == 0:
                 print(f"    Batch {i+1} | Main: {loss_class.item():.3f} | AuxST: {loss_st.item():.3f} | AuxGeo: {loss_geo.item():.3f} | Site: {loss_site.item():.3f}")

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        val_acc, val_f1, _, _ = validate_subject_wise(model, X_val, y_val, groups_val, class_criterion, device)
        
        # scheduler.step(val_acc) # Removed for Cosine
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_asd_model_dann.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    print(f"\n训练结束. 最佳 Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
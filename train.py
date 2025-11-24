import torch 
from evaluation import eval_model
# 【新增】导入 F 用于计算余弦相似度
import torch.nn.functional as F

# --- 【新增】从 gbm_ori 复制 InfoNCE 超参数 ---
CONTRASTIVE_WEIGHT = 0.1 
TEMPERATURE = 0.1


# --- 新增辅助函数 ---
def calculate_tv_loss(node_features, edge_index):
    """计算给定节点特征和边索引的总变差损失"""
    row, col = edge_index
    # 确保图有边，避免除以零
    if row.size(0) == 0:
        return torch.tensor(0.0, device=node_features.device)

    diff = node_features[row] - node_features[col]
    tv_loss = torch.pow(diff, 2).sum() / row.size(0)
    return tv_loss
# --- 新增结束 ---


def info_nce_loss(features_1, features_2, temperature):
    """
    Calculates the InfoNCE loss for contrastive learning.
    (代码来自 gbm_ori/train.py)
    """
    # Normalize the feature vectors to compute cosine similarity
    features_1 = F.normalize(features_1, dim=1)
    features_2 = F.normalize(features_2, dim=1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = torch.matmul(features_1, features_2.T) / temperature
    
    # The labels are the diagonal elements
    labels = torch.arange(features_1.shape[0], device=features_1.device)
    
    # The loss is the cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss
# --- 【新增结束】 ---


def train(model, train_loader, test_loader, loss_func, optimizer, scheduler, args):
    num_epochs = args.num_epochs
    writer = args.writer
    logger = args.logger
    device = args.device

    # 【新增】定义正则化项和对比损失的权重超参数
    lambda_tv = 0       # 总变差损失权重
   
    
    model.to(device)
    round = 0
    best_test_auc = 0.0
    for epoch in range(num_epochs):
        model.train()        
        total_loss_ = 0.0
        clf_loss_ = 0.0
        tv_loss_ = 0.0
        cl_loss_ = 0.0 # 【新增】用于记录对比损失
        batch_count = 0.0
        for data in train_loader:
            
            data = data.to(device)
            y = data.y
            y = y.squeeze().long()
            
            y_pred, (coord_node_features, coord_edge_index), (feature_node_features, feature_edge_index), coord_proj, feature_proj = model(data)
# --- 修改结束 ---

            classification_loss = loss_func(y_pred, y)

            tv_loss_coord = calculate_tv_loss(coord_node_features, coord_edge_index)
            tv_loss_feature = calculate_tv_loss(feature_node_features, feature_edge_index)


            tv_loss = tv_loss_coord + tv_loss_feature

            contrastive_loss = info_nce_loss(coord_proj, feature_proj, TEMPERATURE)
            
            total_loss = classification_loss + lambda_tv * tv_loss + CONTRASTIVE_WEIGHT * contrastive_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (scheduler is not None) and (args.scheduler not in  ["MULTILR", "ExponentialLR"]):
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[0]
                writer.add_scalar("LR", learning_rate, round)
                round += 1

            total_loss_ += total_loss.item()
            clf_loss_ += classification_loss.item()
            tv_loss_ += tv_loss.item()
            cl_loss_ += contrastive_loss.item() 
            batch_count += 1
 
        if args.scheduler == "MULTILR" or args.scheduler == "ExponentialLR":
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]
            writer.add_scalar("LR", learning_rate, epoch+1)

       # 【修改】使用 val_loader 进行评估
        eval_model_wrapper = lambda m, d: m(d)[0]
        eval_train = eval_model(model, train_loader, device, model_wrapper=eval_model_wrapper)
        eval_test = eval_model(model, test_loader, device, model_wrapper=eval_model_wrapper)
        current_test_auc = eval_test[1]

        avg_total_loss = total_loss_ / batch_count


        logger.info("Epoch[%d/%d], Total Loss:%.4f (CLF:%.4f, TV:%.4f, CL:%.4f)" % 
                    (epoch+1, num_epochs, total_loss_ / batch_count, 
                     clf_loss_ / batch_count, tv_loss_ / batch_count, cl_loss_ / batch_count))
        
        logger.info("Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[1], eval_train[0], eval_train[2],
                       eval_test[1], eval_test[0], eval_test[2]))
        
        """tensorboard"""
        
        writer.add_scalar("LOSS/Total", total_loss_ / batch_count, epoch + 1)
        writer.add_scalar("LOSS/Classification", clf_loss_ / batch_count, epoch + 1)
        writer.add_scalar("LOSS/TotalVariation", tv_loss_ / batch_count, epoch + 1)
        writer.add_scalar("LOSS/Contrastive", cl_loss_ / batch_count, epoch + 1) # 【新增】
        
        writer.add_scalars("AUC", {"AUC_train": eval_train[1], "AUC_test": eval_test[1]}, epoch + 1)
        
        if args.savedisk and (current_val_auc > best_val_auc):
            best_val_auc = current_val_auc
            logger.info(f"*** New best model on VAL set! Val AUC: {best_val_auc:.4f}, saving... ***")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": avg_total_loss,
                "test_auc": best_test_auc, # 【新增】同时保存最佳分数
                "NUM_EPOCHS": num_epochs,
                "DEVICE": device
            }

            torch.save(state, "{}/model.pt".format(args.model_path))
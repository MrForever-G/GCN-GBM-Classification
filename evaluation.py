import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch


def ensemble_patient(patientID, y_true, y_probs):
    # patientID: a list with patient id
    # y_probs: prediction score of a sample sample_num x class_num
    # y_true: ground-truth label of a sample
    uq_patientID = list(set(patientID))

    ensemble_y_probs = []
    ensemble_y_true = []

    for uq_id in uq_patientID:
        bool_idx = [True if uq_id == id else False for id in patientID]
        ensemble_y_probs.append(y_probs[bool_idx, :].mean(axis=0))
        ensemble_y_true.append(y_true[bool_idx].mean())

    ensemble_y_true = np.array(ensemble_y_true).ravel()
    ensemble_y_probs = np.array(ensemble_y_probs)

    return uq_patientID, ensemble_y_probs, ensemble_y_true


def auc(target, score):
    # 确保 target 是整数类型
    target = target.astype(int)
    if len(set(target)) > 2:
        auc_ = roc_auc_score(target, score, average="macro", multi_class="ovr")
    elif len(set(target)) == 2:
        # 对于二分类，score 应该是正类的概率
        auc_ = roc_auc_score(target, score[:, 1])
    else:
        # 如果只有一个类别，AUC没有意义，返回0.5
        return 0.5
    return auc_


    
def full_evaluation(y_true, y_probs):
    result = bootstrap_ap(y_true, y_probs, 1000, 0.95)
    return result


def eval_model(model, loader, device, model_wrapper=None):
    y_true = []
    y_probs = []
    patientID = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            patientID_ = data.sample_id
            data = data.to(device)
            y = data.y
            
            # ==========================================================
            # 【修改点 2】: 使用 if/else 逻辑来调用模型，实现解耦
            # ==========================================================
            if model_wrapper:
                # 如果提供了 wrapper，就用它来获取模型的分类输出
                linear_prob = model_wrapper(model, data)
            else:
                # 否则，假设模型直接返回分类输出
                linear_prob = model(data)
            
            prob = torch.softmax(linear_prob, dim=1)
            
            y_probs.append(prob)
            
            y_true.append(y)
            patientID += patientID_

    y_true = torch.cat(y_true, dim=0).cpu().numpy().ravel()
    y_probs = torch.cat(y_probs, dim=0).cpu().numpy()
    
    _, ensemble_y_probs, ensemble_y_true = ensemble_patient(patientID, y_true, y_probs)
    metric = full_evaluation(ensemble_y_true, ensemble_y_probs)

    return metric


def bootstrap_ap(target, score, B, c):
    n = len(target)
    sample_result_arr_auc = []
    count = 0
    while True:
        index_arr = np.random.randint(0, n, size=n)
        target_sample = target[index_arr]
        if len(set(target_sample)) == 1:
            continue
        
        score_sample = score[index_arr]
        
        sample_result_auc = auc(target_sample, score_sample)
        sample_result_arr_auc.append(sample_result_auc)
        
        count += 1
        if count > B:
            break


    a = 1 - c
    k1 = int(count * a / 2)
    k2 = int(count * (1 - a / 2))
    ap_sample_arr_auc_sorted = sorted(sample_result_arr_auc)
    auc_lower = ap_sample_arr_auc_sorted[k1]
    auc_higher = ap_sample_arr_auc_sorted[k2]
    auc_mid = ap_sample_arr_auc_sorted[int(count/2)]


    return tuple([auc_lower, auc_mid, auc_higher])
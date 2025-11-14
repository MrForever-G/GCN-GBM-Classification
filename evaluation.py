import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score
import torch
import pandas as pd
import time 

def save_record(patchID, y_true, y_probs):
    record_dict = dict()
    record_dict["patchID"] = patchID
    record_dict["y_true"] = y_true
    for i in range(y_probs.shape[1]):
        record_dict["y_probs%d" % i] = y_probs[:, i]
    record_df = pd.DataFrame.from_dict(record_dict)

    return record_df

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
    if len(set(target)) > 2:
        auc_ = roc_auc_score(target, score, average="macro", multi_class="ovr")
    else:
        auc_ = roc_auc_score(target, score[:, 1])
    return auc_


def get_prec_sen_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    return precision, sensitivity, specificity


def full_evaluation(y_true, y_probs):
    result = bootstrap_ap(y_true, y_probs, 1000, 0.95)
    return result


def eval_model(model, loader, device):
    y_true = []
    y_probs = []
    patientID = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            patientID_ = data.sample_id
            data = data.to(device)
            y = data.y
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
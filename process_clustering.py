import numpy as np
import os 
from sklearn.cluster import KMeans
import torch

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

parent_dir = "./GBM/dinov2_feature"
to_dir = "./GBM/postdata/adaptive_kmeans/"
dir_list = os.listdir(parent_dir)
RANDOM_RUN = 100
check_dir(to_dir)

def perform_kmeans_clustering(corrd, feature, random_state, n_clusters=5):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(corrd)
    labels = kmeans.labels_
    
    center_corrd = kmeans.cluster_centers_
    center_feature = []
    for i in range(n_clusters):
        cluster_features = feature[labels == i]
        center_feature.append(np.mean(cluster_features, axis=0))
    center_feature = np.array(center_feature)
    
    return labels, center_corrd, center_feature


for dir_name in dir_list:
    dir_path = os.path.join(parent_dir, dir_name)
    corrd = []
    feature = []
    file_name = list(os.listdir(dir_path))
    for file_name_ in os.listdir(dir_path):
        if file_name_.endswith(".npy"):
            sample_id = file_name_.split(".npy")[0]
            corrd_ = np.array([float(sample_id.split("_")[-2]), float(sample_id.split("_")[-1])])
            corrd.append(corrd_)
            feature_ = np.load(os.path.join(dir_path, file_name_))
            feature.append(feature_)
            
    corrd = np.array(corrd)
    feature = np.array(feature) 
    
    patch_num = corrd.shape[0]
    print(f"Processing {dir_name}, patch num: {patch_num}")
    
    if patch_num < 200:
        print("Skip keams for small patch num")
        center_corrd = corrd
        center_feature = feature
        labels = -1 * np.ones(patch_num)
        file_path = os.path.join(to_dir, f"{dir_name}_rpt_none.pt")
        torch.save({'labels': torch.from_numpy(labels), 
                    'center_corrd': torch.from_numpy(center_corrd), 
                    'center_feature': torch.from_numpy(center_feature), 
                    'file_name': file_name}, file_path)
        continue
    elif patch_num < 1000:
        n_clusters = 200
    else:
        n_clusters = int(patch_num // (np.sqrt(patch_num) / np.sqrt(np.log10(patch_num))))  
    
    for run in range(RANDOM_RUN):
        random_state = run * 100 + 2025
        labels, center_corrd, center_feature = perform_kmeans_clustering(corrd, feature, random_state, n_clusters)
        file_path = os.path.join(to_dir, f"{dir_name}_rpt_{run}.pt")
        torch.save({'labels': torch.from_numpy(labels), 
                    'center_corrd': torch.from_numpy(center_corrd), 
                    'center_feature': torch.from_numpy(center_feature), 
                    'file_name': file_name}, file_path)
        print(f"Shape of center_feature: {center_feature.shape}, saved to {file_path}")

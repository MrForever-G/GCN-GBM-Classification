import numpy as np 
import os 
import torch
from utils import check_dir
from sklearn.neighbors import NearestNeighbors

def knn_connectivity(features, k=5, sigma=0.5):

    n_samples = features.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(features)
   
    distances, indices = nbrs.kneighbors(features)
    # drop self (first column)
    indices = indices[:, 1:]

    # adjacency initialization
    A = np.zeros((n_samples, n_samples), dtype=int)
    W = np.zeros((n_samples, n_samples), dtype=float)
    # fill adjacency (single loop)
    for i in range(n_samples):
        A[i, indices[i]] = 1
        W[i, indices[i]] = distances[i, 1:]

    # symmetrize to make undirected graph
    A = np.maximum(A, A.T)
  

    local_distances = distances.sum(axis=1) / k
    eps = local_distances.reshape(n_samples, 1) + local_distances.reshape(1, n_samples)
    weights = np.exp(-W ** 2 / ((sigma * eps) ** 2))

    weights = (weights + weights.T) / 2
    weights = weights * A  # ensure weights are zero where there is no edge
    
    # convert to for COO format, PyTorch Geometric style
    row, col = np.nonzero(A)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    edge_weight = torch.tensor(weights[row, col], dtype=torch.float)
   
    return edge_index, edge_weight

parent_dir = "./GBM/postdata/adaptive_kmeans/"
to_dir = "./GBM/postdata/graph_structure/"
check_dir(to_dir)
dir_list = os.listdir(parent_dir)
file_name = [f for f in os.listdir(parent_dir) if f.endswith(".pt")]


for file_name_ in file_name:
    data = torch.load(os.path.join(parent_dir, file_name_))
    sample_id = file_name_.split("_")[0]
    center_corrd = data['center_corrd'].numpy()
    center_feature = data['center_feature'].numpy()
    
    node_num = center_feature.shape[0]
    # adaptive knn number
    knearest_num = max(5, np.ceil(np.log2(node_num)).astype(int))
    
    print(f"Processing {file_name_}, patch num: {center_corrd.shape[0]}")
    
    corrd_edge_index, corrd_edge_weight = knn_connectivity(center_corrd, k=knearest_num, sigma=0.5)
    feature_edge_index, feature_edge_weight = knn_connectivity(center_feature, k=knearest_num, sigma=0.5)
    
    
    file_path = os.path.join(to_dir, f"{file_name_}_adj.pt")
    
    torch.save({'center_corrd': torch.from_numpy(center_corrd), 
                'center_feature': torch.from_numpy(center_feature),
                'corrd_edge_index': corrd_edge_index,
                'corrd_edge_weight': corrd_edge_weight,
                'feature_edge_index': feature_edge_index,
                'feature_edge_weight': feature_edge_weight,
                'sample_id': sample_id}, 
                file_path)


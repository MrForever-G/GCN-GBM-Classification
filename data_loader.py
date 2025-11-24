import pandas as pd
import torch
import os   
import numpy as np
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj

class GBMGraphDataset(Dataset):
    def __init__(self, data_dir, sample_info_csv, is_train=False):
        super().__init__()
        self.data_dir = data_dir
        self.sample_info = pd.read_csv(sample_info_csv)
        
        self.label_type = {'Classical': 0, 'Mesenchymal':1, 'Neural': 2, 'Proneural': 3}
        
        # Create a mapping from sampleID to label
        self.sample_to_label = {row['sampleID']: self.label_type[row['GeneExp_Subtype']] 
                                for _, row in self.sample_info.iterrows()}
        
        # Create a mapping from sampleID to days_to_last_followup
        self.sample_to_followup = {row['sampleID']: row['days_to_last_followup'] 
                                   for _, row in self.sample_info.iterrows()}
        
        # Create a mapping from sampleID to days_to_death
        self.sample_to_death = {row['sampleID']: row['days_to_death'] 
                                for _, row in self.sample_info.iterrows()}
        
        # List all .pt files in the directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.is_train = is_train
        
    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        pt_file = self.file_list[idx]
        sample_id = pt_file.split('_')[0]  # Assuming filename format is <sampleID>_something.pt
        pt_path = os.path.join(self.data_dir, pt_file)
        
        # Load the .pt file
        data = torch.load(pt_path, weights_only=False)
        
        # Get the label for the sample
        label = self.sample_to_label.get(sample_id, -1)  # Default to -1 if not found
        
        days_to_last_followup = self.sample_to_followup.get(sample_id, np.nan)
        days_to_death = self.sample_to_death.get(sample_id, np.nan)
        
        if label == -1:
            raise ValueError(f"Label for sample ID {sample_id} not found.")
        
        if days_to_last_followup is not np.nan and days_to_death is not np.nan:
            survival_time = min(days_to_last_followup, days_to_death)
            censorship = int(days_to_death <= days_to_last_followup)
        elif days_to_last_followup is not np.nan:
            survival_time = days_to_last_followup   
            censorship = 0    
        elif days_to_death is not np.nan:
            survival_time = days_to_death
            censorship = 1
        else:
            survival_time = np.nan
            censorship = np.nan
        
        if censorship is np.nan or survival_time is np.nan:
            raise ValueError(f"Survival data for sample ID {sample_id} is incomplete.")
        
        # ==========================================================
        # 【新增】数据增强逻辑块
        # ==========================================================
        if self.is_train:
            # --- 增强方法1: 节点特征掩码 (Node Feature Masking) ---
            # 定义掩码概率，例如 15%
            mask_prob = 0.15
            # 创建一个与特征矩阵同样形状的随机数矩阵，值在 [0, 1) 之间
            mask = torch.rand(data['center_feature'].shape) < mask_prob
            # 将掩码为 True 的位置的特征置为 0
            data['center_feature'][mask] = 0.0

            # --- 增强方法2: 边丢弃 (Edge Dropout) ---
            # 定义丢弃边的概率，例如 20%
            edge_dropout_p = 0.2
            # 对空间图进行边丢弃
            data['corrd_edge_index'], data['corrd_edge_weight'] = dropout_adj(
                edge_index=data['corrd_edge_index'], 
                edge_attr=data['corrd_edge_weight'], 
                p=edge_dropout_p, 
                force_undirected=True, # 保持图的无向性
                training=self.is_train# 确保只在训练模式下生效
            )
            # 对特征图进行边丢弃
            data['feature_edge_index'], data['feature_edge_weight'] = dropout_adj(
                edge_index=data['feature_edge_index'], 
                edge_attr=data['feature_edge_weight'], 
                p=edge_dropout_p, 
                force_undirected=True,
                training=self.is_train
            )
        # ==========================================================
        # 数据增强结束
        # ==========================================================
        
        g = HeteroData()
        g["corrd"].x = data["center_corrd"]
        g["feature"].x = data['center_feature']
        g["corrd", "to", "corrd"].edge_index = data['corrd_edge_index']
        g["corrd", "to", "corrd"].edge_attr = data['corrd_edge_weight']
        g["feature", "to", "feature"].edge_index = data['feature_edge_index']
        g["feature", "to", "feature"].edge_attr = data['feature_edge_weight']
        g.y = torch.tensor([label], dtype=torch.long)
        g.survival_time = torch.tensor([survival_time], dtype=torch.float)
        g.censorship = torch.tensor([censorship], dtype=torch.long)
        g.sample_id = sample_id
       
        return g
    

# # Example usage
# if __name__ == "__main__":
#     dataset = GBMGraphDataset(data_dir="E:/data factory/GBM/postdata/train_test_split/train", 
#                               sample_info_csv="E:/data factory/GBM/clinical_data_processed.csv")
    
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     for batch in loader:
#         print(batch)
#         print(batch.batch)
#         print(f"Batch size: {batch.num_graphs}")
#         print(f"Node features shape: {batch["corrd"].x.shape}")
#         print(f"Edge index shape: {batch["corrd", "to", "corrd"].edge_index.shape}")
#         print(f"Labels shape: {batch.y.shape}")
#         print(f"Sample IDs: {batch.sample_id}")
#         print(f"Survival times: {batch.survival_time}")
#         print(f"Censorships: {batch.censorship}")
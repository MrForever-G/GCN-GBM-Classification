import pandas as pd
import torch
import os   
import numpy as np
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader

class GBMGraphDataset(Dataset):
    def __init__(self, data_dir, sample_info_csv):
        super().__init__()
        self.data_dir = data_dir
        self.sample_info = pd.read_csv(sample_info_csv)
        self.label_type = {'Proneural': 0, 'Mesenchymal': 1}
        self.sample_info = self.sample_info[self.sample_info["GeneExp_Subtype"].isin(self.label_type.keys())].copy()
        self.sample_info["sampleID"] = self.sample_info["sampleID"].astype(str)

        
        # Create a mapping from sampleID to label
        self.sample_to_label = dict(zip(self.sample_info["sampleID"], 
                                self.sample_info["GeneExp_Subtype"].map(self.label_type)))

        # Create a mapping from sampleID to days_to_last_followup
        self.sample_to_followup = {row['sampleID']: row['days_to_last_followup'] 
                                   for _, row in self.sample_info.iterrows()}
        
        # Create a mapping from sampleID to days_to_death
        self.sample_to_death = {row['sampleID']: row['days_to_death'] 
                                for _, row in self.sample_info.iterrows()}
        
        # List all .pt files in the directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        
        
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
        batch = torch.zeros(data["center_corrd"].size(0), dtype=torch.long) + idx
        g.batch = batch
        
        return g
    

# Example usage
if __name__ == "__main__":
    dataset = GBMGraphDataset(data_dir="E:/data factory/GBM/postdata/train_test_split/train", 
                              sample_info_csv="E:/data factory/GBM/clinical_data_processed.csv")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        print(batch)
        print(batch.batch)
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch['corrd'].x.shape}")
        print(f"Edge index shape: {batch['corrd', 'to', 'corrd'].edge_index.shape}")
        print(f"Labels shape: {batch.y.shape}")
        print(f"Sample IDs: {batch.sample_id}")
        print(f"Survival times: {batch.survival_time}")
        print(f"Censorships: {batch.censorship}")
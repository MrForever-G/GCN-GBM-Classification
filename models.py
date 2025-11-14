import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Set2Set, global_mean_pool
from data_loader import GBMGraphDataset

class Set2SetPoolNet(torch.nn.Module):
    def __init__(self, in_channels, heads, out_channels=128):
        super(Set2SetPoolNet, self).__init__()
        
        # Attention-based diffusion layer
        # self.attention_linear = torch.nn.Linear(in_channels, heads)
        # self.attention_softmax = torch.nn.Softmax(dim=1)
        
        # Graph convolution layers
        self.conv1 = GCNConv(in_channels, 512)
        self.ln1 = torch.nn.LayerNorm(512)
        self.conv2 = GCNConv(512, 256)
        self.ln2 = torch.nn.LayerNorm(256)
        
        # Aggregation layer
        # self.set2set = Set2Set(32, 2)  
        self.aggreation = global_mean_pool
        
        # Final fully connected layer for classification/regression
        self.fc = torch.nn.Linear(256, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # TBD attention diffusion layer
        # x = self.attention_linear(x)
        # x1 = self.attention_softmax(x)
        
        
        # Graph convolution layers
        x2 = self.conv1(x, edge_index, edge_weight)
        x2 = torch.relu(x2)
        x2 = self.ln1(x2)
        
        x3 = self.conv2(x2, edge_index, edge_weight)
        x3 = torch.relu(x3)
        x3 = self.ln2(x3)
        
        # keep node-level features and edges for TV (before pooling)
        self.last_node_features = x3                # [N, 256]
        self.last_edge_index = edge_index           # [2, E]
        self.last_batch = data.batch                # if you need per-graph info later

        # Set2Set layer for graph-level vector
        # x4 = self.set2set(x3, data.batch)
        x4 = self.aggreation(x3, data.batch)
        
        
        # Final fully connected layer
        x4 = self.fc(x4)
        
        return x4

class TwinsGCN(torch.nn.Module):
    def __init__(self, in_channels, heads, out_channels, class_num, edge_mode="both"):
        super(TwinsGCN, self).__init__()
        self.coord_gcn = Set2SetPoolNet(in_channels, heads, out_channels)
        self.feature_gcn = Set2SetPoolNet(in_channels, heads, out_channels)
        self.fc_final = torch.nn.Linear(out_channels * 2, class_num)
        self.edge_mode = edge_mode.lower()

    def forward(self, data):
        
        unique_graphs = torch.unique(data.batch)
        # Re-index batch to have consecutive graph indices
        for i in range(len(unique_graphs)):
            mask = (data.batch == unique_graphs[i])
            data.batch[mask] = i
            
        coord_data = Data(x=data["feature"].x, edge_index=data["corrd", "to", "corrd"].edge_index, 
                          edge_attr=data["corrd", "to", "corrd"].edge_attr, batch=data.batch)
        feature_data = Data(x=data["feature"].x, edge_index=data["feature", "to", "feature"].edge_index, 
                            edge_attr=data["feature", "to", "feature"].edge_attr, batch=data.batch)
        
        if self.edge_mode == "feature":
            feature_out = self.feature_gcn(feature_data)
            coord_out = torch.zeros_like(feature_out)
        elif self.edge_mode == "corrd":
            coord_out = self.coord_gcn(coord_data)
            feature_out = torch.zeros_like(coord_out)
        else:  # both
            coord_out = self.coord_gcn(coord_data)
            feature_out = self.feature_gcn(feature_data)

        # build TV context for this batch
        self.tv_context = []
        if self.edge_mode in ["corrd", "both"]:
            self.tv_context.append((self.coord_gcn.last_node_features,
                                    self.coord_gcn.last_edge_index))
        if self.edge_mode in ["feature", "both"]:
            self.tv_context.append((self.feature_gcn.last_node_features,
                                    self.feature_gcn.last_edge_index))
            
        # Combine outputs from both GCNs
        combined = torch.cat([coord_out, feature_out], dim=1)
        out = self.fc_final(combined)
        return out

# Example usage
if __name__ == "__main__":
    dataset = GBMGraphDataset(data_dir="./GBM/postdata/train_test_split/train", 
                              sample_info_csv="./GBM/clinical_data_processed.csv")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        print(batch)
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch['corrd'].x.shape}")
        print(f"Edge index shape: {batch['corrd', 'to', 'corrd'].edge_index.shape}")
        print(f"Labels shape: {batch.y.shape}")
        print(f"Sample IDs: {batch.sample_id}")
        model = TwinsGCN(in_channels=batch['feature'].x.shape[1], heads=32, out_channels=4)
        out = model(batch)
        print(f"Model output shape: {out.shape}")
        break
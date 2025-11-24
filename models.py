import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Set2Set, global_mean_pool
from data_loader import GBMGraphDataset
import torch.nn.functional as F


class CoordGCN(torch.nn.Module):
    def __init__(self, in_channels, heads, out_channels, dropout_rate=0.5):
        super(CoordGCN, self).__init__()
        
        # 1. 先使用线性层将 2D 映射到 heads (e.g., 2 -> 8)
        self.attention_linear = torch.nn.Linear(in_channels, heads)
        self.attention_softmax = torch.nn.Softmax(dim=1)
        
        # 2. GCN 在 heads 维度上操作 (e.g., 8 -> 16 -> 32)
        self.conv1 = GCNConv(heads, 16)
        self.ln1 = torch.nn.LayerNorm(16)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        
        self.conv2 = GCNConv(16, 32)
        self.ln2 = torch.nn.LayerNorm(32)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        # 3. 池化
        self.aggreation = global_mean_pool
        
        # 4. 最终 FC (32 -> out_channels)
        self.fc = torch.nn.Linear(32, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 应用 gbm_ori 的预处理
        x = self.attention_linear(x)
        x1 = self.attention_softmax(x)
        
        x2 = self.conv1(x1, edge_index, edge_weight) # 注意这里用 x1
        x2 = torch.relu(x2)
        x2 = self.ln1(x2)
        x2 = self.dropout1(x2)

        x3 = self.conv2(x2, edge_index, edge_weight)
        x3 = torch.relu(x3)
        x3 = self.ln2(x3)
        x3 = self.dropout2(x3)

        x4 = self.aggreation(x3, data.batch)
        x4 = self.fc(x4)
        
        # 返回 (节点特征, 图特征)
        return x3, x4

# ==========================================================
# 【修改点 4】: 创建 FeatureGCN (基于 gbm_tea 的 "wide" 设计)
# 这个 GCN 用于处理 768D 特征
# ==========================================================
class FeatureGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(FeatureGCN, self).__init__()
        
        # 1. GCN 使用 gbm_tea 的 "wide" 架构 (e.g., 768 -> 512 -> 256)
        self.conv1 = GCNConv(in_channels, 512)
        self.ln1 = torch.nn.LayerNorm(512)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        
        self.conv2 = GCNConv(512, 256)
        self.ln2 = torch.nn.LayerNorm(256)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        # 2. 池化
        self.aggreation = global_mean_pool
        
        # 3. 最终 FC (256 -> out_channels)
        self.fc = torch.nn.Linear(256, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # 直接应用 GCN (没有 attention_linear)
        x2 = self.conv1(x, edge_index, edge_weight)
        x2 = torch.relu(x2)
        x2 = self.ln1(x2)
        x2 = self.dropout1(x2)

        x3 = self.conv2(x2, edge_index, edge_weight)
        x3 = torch.relu(x3)
        x3 = self.ln2(x3)
        x3 = self.dropout2(x3)

        x4 = self.aggreation(x3, data.batch)
        x4 = self.fc(x4)
        
        # 返回 (节点特征, 图特征)
        return x3, x4



        
class TwinsGCN(torch.nn.Module):
    # 【修改点 3】: 修改 __init__ 函数签名，接收两个独立的输入维度
    def __init__(self, coord_in_channels, feature_in_channels, heads, out_channels, class_num):
        super(TwinsGCN, self).__init__()
        
        # 【修改点 4】: 为每个分支传入各自正确的输入维度
        self.coord_gcn = CoordGCN(coord_in_channels, heads, out_channels)
        self.feature_gcn = FeatureGCN(feature_in_channels, out_channels)
        self.final_dropout = torch.nn.Dropout(p=0.5)
        self.fc_final = torch.nn.Linear(out_channels*2, class_num)

        projection_dim = 64
        self.coord_projector = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, projection_dim)
        )
        self.feature_projector = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, projection_dim)
        )
        

    def forward(self, data):
      
        coord_x = data["corrd"].x.float()
        feature_x = data["feature"].x.float()
            
        coord_data = Data(x=coord_x, edge_index=data["corrd", "to", "corrd"].edge_index, 
                          edge_attr=data["corrd", "to", "corrd"].edge_attr, batch=data["corrd"].batch)
        feature_data = Data(x=feature_x, edge_index=data["feature", "to", "feature"].edge_index, 
                            edge_attr=data["feature", "to", "feature"].edge_attr, batch=data["feature"].batch)

        # coord_out 和 feature_out 现在都是一个元组 (node_features, graph_features)
        coord_node_features, coord_graph_out = self.coord_gcn(coord_data)
        feature_node_features, feature_graph_out = self.feature_gcn(feature_data)

        coord_proj = self.coord_projector(coord_graph_out)
        feature_proj = self.feature_projector(feature_graph_out)

        combined = torch.cat([coord_graph_out, feature_graph_out], dim=1)
       
        combined_with_dropout = self.final_dropout(combined)
        out = self.fc_final(combined_with_dropout)
       
        coord_branch_out = (coord_node_features, coord_data.edge_index)
        feature_branch_out = (feature_node_features, feature_data.edge_index)

        return out, coord_branch_out, feature_branch_out, coord_proj, feature_proj


# Example usage (本地测试代码部分无需修改)
if __name__ == "__main__":
    # 此部分仅用于本地测试，在实际运行时不会被调用
    # 为了让它能运行，需要模拟数据结构
    
    # 模拟一个 batch 数据
    num_nodes_coord = 100
    num_nodes_feature = 100
    coord_x = torch.rand((num_nodes_coord, 2)) # 2维坐标
    feature_x = torch.rand((num_nodes_feature, 768)) # 768维特征
    
    # 创建模拟的异构图数据对象
    from torch_geometric.data import HeteroData
    batch = HeteroData()
    batch['corrd'].x = coord_x
    batch['feature'].x = feature_x
  
    print("代码结构已修复。请在您的 run.py 中进行相应修改后运行。")
   
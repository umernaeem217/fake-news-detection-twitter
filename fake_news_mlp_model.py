import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

class FakeNewsMLPModel(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_ratio=0.0, pool_type='mean'):
        super(FakeNewsMLPModel, self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.pool_type = pool_type

        # A simple MLP: Input -> Hidden -> Output
        # You can add more layers or batch normalization if you want
        self.fc1 = torch.nn.Linear(self.num_features, self.nhid)
        self.fc2 = torch.nn.Linear(self.nhid, self.num_classes)
        self.dropout = torch.nn.Dropout(self.dropout_ratio)

    def forward(self, data):
        # data.x: Node feature matrix for the entire batch
        # data.batch: Assignment of each node to a graph
        x, batch = data.x, data.batch

        # Pool node features for each graph into a single graph-level embedding.
        # We can try either mean or max pooling.
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_max_pool(x, batch)

        # Apply a simple two-layer MLP
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # Use log_softmax for consistency with the graph model
        x = F.log_softmax(x, dim=-1)
        return x

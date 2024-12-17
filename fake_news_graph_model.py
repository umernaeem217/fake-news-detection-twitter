import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class FakeNewsGraphModel(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_ratio=0.0, model_type='sage', concat=False):
        super(FakeNewsGraphModel, self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.model_type = model_type
        self.concat = concat

        if self.model_type == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model_type == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model_type == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)
        else:
            raise ValueError("Invalid model type. Choose from 'gcn', 'sage', 'gat'.")

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)

        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        x = F.log_softmax(self.lin2(x), dim=-1)

        return x
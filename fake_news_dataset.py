from torch_geometric.data import InMemoryDataset
import os.path as osp
import torch
from utils.graph_utility import read_graph_data

VALID_DATASETS = {'politifact', 'gossipcop'}
VALID_FEATURES = {'profile', 'content', 'bert'}

class FakeNewsDataSet(InMemoryDataset):
    def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
        if name not in VALID_DATASETS:
            raise ValueError(f"Invalid dataset name '{name}'. Valid options are: {VALID_DATASETS}")
        if feature not in VALID_FEATURES:
            raise ValueError(f"Invalid feature '{feature}'. Valid options are: {VALID_FEATURES}")
        self.name = name
        self.root = root
        self.feature = feature
        super(FakeNewsDataSet, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        names = ['node_graph_id', 'graph_labels']
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

    def download(self):
        pass

    def process(self):

        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
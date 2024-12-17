from torch_geometric.utils import to_undirected
from torch_sparse import coalesce

class UndirectedTransformer:
	def __init__(self):
		pass

	def __call__(self, data):
		edge_attr = None
		edge_index = to_undirected(data.edge_index, data.x.size(0))
		num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
		edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
		data.edge_index = edge_index
		data.edge_attr = edge_attr
		return data
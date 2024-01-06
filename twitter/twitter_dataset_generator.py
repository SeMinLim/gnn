import torch
import torch_geometric.transforms as T

data = torch.load("/mnt/ephemeral/gnn/dataset/Twitter/twitter.pt")
data = T.ToUndirected()(data)
print('The current data is undirected:', data.is_undirected())
torch.save(data, "/mnt/ephemeral/gnn/dataset/Twitter/twitter.pt")

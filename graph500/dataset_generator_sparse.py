import struct
import torch
import torch_geometric
import torch_geometric.transforms as T
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor


# Node setting
numNodes = 8388608 #S23 


# PyTorch availability and GPU checking
print('PyTorch Availability:', torch.cuda.is_available(), flush=True)
print('The Number of GPUs:', torch.cuda.device_count(), flush=True)
print('The Current GPU Index:', torch.cuda.current_device(), flush=True)
print('The Current GPU Name:', torch.cuda.get_device_name(0), flush=True)


# Read the file and store the dataset
file = open("/mnt/ephemeral/gnn/dataset/Graph500/s23ef64/graph500_scale23_ef64.edges", "rb")

edges = []
while True:
    value = file.read(8)
    if len(value) != 8:
        break
    else:
        (v1, v2) = struct.unpack("II", value)
        edges.append(v1)
        edges.append(v2)

file.close()


## Edges
# Change edge_index list to numpy array first
edges = np.array(edges)
edges = edges.reshape(-1, 2)
print('2D Directed Edge Array:', edges.shape, flush=True)

# Delete duplicates of the edges (It's meaningless having the duplicated edges)
edges = np.unique(edges, axis=0)
print('2D Directed Edge Array:', edges.shape, flush=True)

# Generate the opposite directional edges first
numEdges = len(edges)
opposite = []
for i in range(numEdges):
    edge = edges[i]
    opposite.append(edge[1])
    opposite.append(edge[0])

opposite = np.array(opposite)
opposite = opposite.reshape(-1, 2)
print('2D Opposite Directed Edges Array:', opposite.shape, flush=True)

# Merge the original and opposite edges
edges = np.append(edges, opposite, axis=0)
print('2D Undirected Edges Array Array:', edges.shape, flush=True)

# Delete duplicates of the edges
edges = np.unique(edges, axis=0)
print('Final 2D Undirected Edges Array:', edges.shape, flush=True)

# Make edge_index as tensor for using it on PyTorch Geometric
# dtype should be torch.int64
edges = torch.tensor(edges, dtype=torch.int64)
edges = edges.t().contiguous()
print('2D Undirected Edges Tensor:', edges.shape, flush=True)

# Change edge_index to sparse
edges = SparseTensor(
        row=edges[0],
        col=edges[1],
        sparse_sizes=(numNodes, numNodes),
        is_sorted=True,
        trust_data=True,)
print(edges, flush=True)


## Node Feature Matrix
# Make node feature matrix by our own
# #nodes x #features(64)
x=[]
tmp = []
for i in range(numNodes):
    for j in range(64):
        r = random.uniform(-2.5, 2.5)
        while r in tmp:
            r = random.uniform(-2.5, 2.5)
        tmp.append(r)
    x.extend(tmp)
    tmp.clear()

# Change node feature matrix(list) to 2d numpy array
x = np.array(x)
x = x.reshape(-1, 64)
print('2D Node Feature Array:', x.shape, flush=True)

# Make node feature matrix as tensor for using it on PyTorch Geometric
# dtype should be torch.float32
x = torch.tensor(x, dtype=torch.float32)
print('2D Node Feature Tensor:', x.shape, flush=True)


## Ground-Truth Labels
# Make ground-truth labels by our own
y=[]
for i in range(numNodes):
    r = random.randrange(0, 64)
    y.append(r)

# Make ground-truth lables as tensor for using it on PyTorch Geometric
# dtype should be torch.int64
y = torch.tensor(y, dtype=torch.int64)
print('1D Ground-Truth Label Tensor:', y.shape, flush=True)


## Save the dataset as .pt
# Make node feature matrix, edge index, ground-truth labels as PyTorch Dataset
data = Data(x=x, y=y, adj_t=edges)
print(data, flush=True)

# Save the data
torch.save(data, "/mnt/ephemeral/gnn/dataset/Graph500/s23ef64/graph500_scale23_ef64_sparse.pt")

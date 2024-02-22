import torch
import torch_geometric
import torch_geometric.transforms as T
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor


# Function for reading dataset file
def read_file_until_tab_newline(file):
    buffer = ''

    returnMode = 0
    while True:
        byte = file.read(1)
        if not byte:
            returnMode = 0
            break
        else:
            if byte == "\t" or byte == "\n":
                returnMode = 1
                break
            buffer += byte

    if (returnMode == 0):
        return False
    else:
        return buffer


# Node setting
numNodes = 41652230


# PyTorch availability and GPU checking
print('PyTorch Availability:', torch.cuda.is_available(), flush=True)
print('The Number of GPUs:', torch.cuda.device_count(), flush=True)
print('The Current GPU Index:', torch.cuda.current_device(), flush=True)
print('The Current GPU Name:', torch.cuda.get_device_name(0), flush=True)


# Read the file and store the dataset
file = open("/mnt/ephemeral/gnn/dataset/Twitter/twitter_rv.net", "r")

edges = []
while True:
    value = read_file_until_tab_newline(file)
    if not value:
        print(f'Storing edge_index finished!')
        break
    else:
        value = int(value)
        edges.append(value)

file.close()


## Edges
# Check the list length and node's max and min
print('The Length of Edge List:', len(edges), flush=True)
print('Max of Nodes:', max(edges), flush=True)
print('Min of Nodes:', min(edges), flush=True)

# Change edge_index list to numpy array first
edges = np.array(edges)
print('1D Edge Array:', edges.shape, flush=True)

# Get the nodes
nodes = np.unique(edges)
nodes.sort()
print('1D Node Array', nodes.shape, flush=True)

# Reshape edge_index numpy array to 2-dimensional
edges = edges.reshape(-1, 2)
print('2D Directed Edge Array (Dense):', edges.shape, flush=True)

# Generate the modified node
m_nodes=[]
for i in range(numNodes):
    m_nodes.append(i)

m_nodes = np.array(m_nodes)
nodes = np.stack((nodes, m_nodes), axis=1)

# Sort edge_index based on column 1
tmp_nodes = edges[:, 0]
edges = edges[tmp_nodes.argsort()]
print('Sorting 2D Directed Edge Array by Column 1 Done!', flush=True)

# Change the original nodes to modified nodes (Phase 1)
num_edges = 1468365182
pointer = 0
for i in range(num_edges):
    for j in range(pointer, numNodes):
        if (edges[i][0] == nodes[j][0]):
            edges[i][0] = nodes[j][1]
            pointer = j
            break
print('Modifying Node Index Column 1 Done!', flush=True)

# Sort edge_index based on column 2
tmp_nodes = edges[:, 1]
edges = edges[tmp_nodes.argsort()]
print('Sorting 2D Directed Edge Array by Column 2 Done!', flush=True)

# Change the original nodes to modified nodes (Phase 2)
pointer = 0
for i in range(num_edges):
    for j in range(pointer, numNodes):
        if (edges[i][1] == nodes[j][0]):
            edges[i][1] = nodes[j][1]
            pointer = j
            break
print('Modifying Node Index Column 2 Done!', flush=True)

# Sort modified edge_index based on column 1 again
tmp_nodes = edges[:, 0]
edges = edges[tmp_nodes.argsort()]
print('Sorting 2D Directed Edge Array by Column 1 Again Done!', flush=True)
print(edges[0], flush=True)
print(edges[1468365181], flush=True)

# Generate the opposite directional edges first
opposite = []
for i in range(1468365182):
    edge = edges[i]
    opposite.append(edge[1])
    opposite.append(edge[0])

opposite = np.array(opposite)
opposite = opposite.reshape(-1, 2)
print('2D Opposite Directed Edges Array (Dense):', opposite.shape, flush=True)

# Merge the original and opposite edges
edges = np.append(edges, opposite, axis=0)
print('2D Undirected Edges Array (Dense):', edges.shape, flush=True)

# Delete duplicates of the edges
edges = np.unique(edges, axis=0)
print('Final 2D Undirected Edges Array (Dense):', edges.shape, flush=True)

# Make edge_index as tensor for using it on PyTorch Geometric
# dtype should be torch.int64
edges = torch.tensor(edges, dtype=torch.int64)
edges = edges.t().contiguous()
print('2D Undirected Edges Tensor (Dense):', edges.shape, flush=True)

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
# 41652230(#nodes) x 64(#features)
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

# Change node feature matrix(list) to numpy array first
x = np.array(x)
print('1D Node Feature Array:', x.shape, flush=True)
# Reshape node feature matrix(numpy array) to 2-dimensional
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


## Train Mask
# Make train mask by our own
train_mask=[]
for i in range(numNodes):
    t = random.choice([True, False])
    train_mask.append(t)

# Make train mask as tensor for using it on PyTorch Geometric
# dtype should be torch.bool
train_mask = torch.tensor(train_mask, dtype=torch.bool)
print('1D Train Mask Tensor:', train_mask.shape, flush=True)


## Save the dataset as .pt
# Make node feature matrix, edge index, ground-truth labels as PyTorch Dataset
data = Data(x=x, y=y, train_mask=train_mask, adj_t=edges)
print(data, flush=True)

# Save the data
torch.save(data, "/mnt/ephemeral/gnn/dataset/Twitter/twitter_rv_sparse.pt")

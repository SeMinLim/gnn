import torch
import torch_geometric
import numpy as np
import random
from torch_geometric.data import Data


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


# PyTorch availability and GPU checking
print('PyTorch Availability:', torch.cuda.is_available())
print('The Number of GPUs:', torch.cuda.device_count())
print('The Current GPU Index:', torch.cuda.current_device())
print('The Current GPU Name:', torch.cuda.get_device_name(0))


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
print('The Length of Edge List:', len(edges))
print('Max of Nodes:', max(edges))
print('Min of Nodes:', min(edges))

# Change edge_index list to numpy array first
edges = np.array(edges)
print('The Shape of Edge List:', edges.shape)
# Reshape edge_index numpy array to 2-dimensional
edges = edges.reshape(-1, 2)
print('The Shape of The Original Edges:', edges.shape)

# Generate the opposite directional edges first
opposite = []
for i in range(1468365182):
    edge = edges[i]
    opposite.append(edge[1])
    opposite.append(edge[0])

opposite = np.array(opposite)
print('The Shape of Primary Opposite Edges List:', opposite.shape)
opposite = opposite.reshape(-1, 2)
print('The Shape of Primary Opposite Edges:', opposite.shape)

# Merge the original and opposite edges
edges = np.append(edges, opposite, axis=0)
print('The Shape of Primary Bidirectional Edges:', edges.shape)

# Delete duplicates of the edges
edges = np.unique(edges, axis=0)
print('The Shape of Final Bidirectional Edges:', edges.shape)

# Make edge_index as tensor for using it on PyTorch Geometric
# dtype should be torch.int64
edges = torch.tensor(edges, dtype=torch.int64)
print('Final Edges:', edges.shape)


## Node Feature Matrix
# Make node feature matrix by our own
# 41652230(#nodes) x 16(#features)
x=[]
tmp = []
for i in range(41652230):
    for j in range(16):
        r = random.uniform(-2.5, 2.5)
        while r in tmp:
            r = random.uniform(-2.5, 2.5)
        tmp.append(r)
    x.extend(tmp)
    tmp.clear()

print('Node Feature List:', len(x))

# Change node feature matrix(list) to numpy array first
x = np.array(x)
print('The Shape of Node Feature List:', x.shape)
# Reshape node feature matrix(numpy array) to 2-dimensional
x = x.reshape(-1, 16)
print('The Shape of Node Feature Matrix:', x.shape)

# Make node feature matrix as tensor for using it on PyTorch Geometric
# dtype should be torch.float32
x = torch.tensor(x, dtype=torch.float32)
print('Node Feature Matrix:', x.shape)


## Ground-Truth Labels
# Make ground-truth labels by our own
y=[]
for i in range(41652230):
    r = random.randrange(0, 16)
    y.append(r)

print('The Shape of Ground-Truth Labels:', len(y))

# Make ground-truth lables as tensor for using it on PyTorch Geometric
# dtype should be torch.int64
y = torch.tensor(y, dtype=torch.int64)
print('Ground-Truth Labels:', y.shape)


## Save the dataset as .pt
# Make node feature matrix, edge index, ground-truth labels as PyTorch Dataset
data = Data(x=x, edge_index=edges.t().contiguous(), y=y)
print('Final PyTorch Dataset:', data)

# Save the data
torch.save(data, "/mnt/ephemeral/gnn/dataset/Twitter/twitter.pt")


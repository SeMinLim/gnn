import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader


# Import Graph500_Scale23_EdgeFactor64
data = torch.load("/mnt/ephemeral/gnn/dataset/Graph500/graph500_scale23_ef64.pt")
data = T.ToSparseTensor()(data)
data = data.pin_memory()


# Single-layer GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(64, 64)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

    def forward(self, x, adj_t):
        x = self.gcn1(x, adj_t)
        z = F.log_softmax(x, dim=1)
        return x, z


# Define required functions
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

def train(model, data):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = model.optimizer
    epochs = 10

    model.train()
    for epoch in range(epochs):
        # Training
        optimizer.zero_grad()
        h, out = model(data.x, data.adj_t)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    return model, h, out


# Do initial setting for training
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
gcn = GCN()


# Sleep for 10 seconds to execute monitoring system
time.sleep(10)
print('Training Start!')


# Training
start.record()
gcn_model, gcn_output, final_output = train(gcn.to('cuda:0'), data.to('cuda:0', non_blocking=True))
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
print('Training Done!')
print('Elapsed Time (100 Epochs):', elapsed_time*0.001, 'seconds')

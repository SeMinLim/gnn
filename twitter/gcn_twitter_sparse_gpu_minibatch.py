import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader


# Import Twitter Dataset
data = torch.load("/mnt/ephemeral/gnn/dataset/Twitter/twitter_sparse.pt")
data = data.pin_memory()


# Single-layer GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(16, 16)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

    def forward(self, x, adj_t):
        x = self.gcn1(x, adj_t)
        z = F.log_softmax(x, dim=1)
        return x, z


# Define required functions
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

def train(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = model.optimizer
    epochs = 5

    model.train()
    for epoch in range(epochs):
        # Training on batches
        for batch in train_loader:
            batch = batch.to('cuda:0', non_blocking=True)
            optimizer.zero_grad()
            h, out = model(batch.x, batch.adj_t)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    return model, h, out


# Do initial setting for training
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
gcn = GCN()


# NeighborLoader
train_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=262144,
        pin_memory=True,
        num_workers=16,
)


# Sleep for 10 seconds to execute monitoring system
time.sleep(5)
print('Training Start!')


# Training
start.record()
gcn_model, gcn_output, final_output = train(gcn.to('cuda:0'), train_loader)
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
print('Training Done!')
print('Elapsed Time (5 Epochs):', elapsed_time*0.001, 'seconds')

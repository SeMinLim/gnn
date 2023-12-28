import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader


# Import Reddit
start_pr = time.process_time()
dataset = Reddit(root="/home/semin/gnn/dataset/Reddit", transform=T.ToSparseTensor())
data = dataset[0]
data = data.pin_memory()


# Single-layer GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(dataset.num_features, dataset.num_classes)
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
    epochs = 100

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
end_pr = time.process_time()
elapsed_time_pr = end_pr - start_pr
print('Model Preparation Done!')
print('Elapsed Time (Preparation):', elapsed_time_pr, 'seconds')


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

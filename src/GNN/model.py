import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
    gt_pw_dif = torch.sub(all_one @ ground_truth.t(), ground_truth @ all_one.t())
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [N, F], adj: [N, N]
        h = torch.matmul(adj, x)
        h = self.linear(h)
        return F.relu(h)

class GNNModel(nn.Module):
    def __init__(self, stocks, time_steps, channels):
        super(GNNModel, self).__init__()
        self.stocks = stocks
        self.time_steps = time_steps
        self.channels = channels
        self.gcn1 = SimpleGCNLayer(channels, 128)
        self.gcn2 = SimpleGCNLayer(128, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs, adj=None):
        # inputs: [stocks, time_steps, channels]
        # For simplicity, use the last time step as node features
        x = inputs[:, -1, :]  # [stocks, channels]
        if adj is None:
            adj = torch.eye(self.stocks, device=x.device)  # Identity if no adj provided
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        out = self.fc(x)  # [stocks, 1]
        return out 
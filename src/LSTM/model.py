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


class LSTMModel(nn.Module):
    def __init__(self, stocks, time_steps, channels, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=channels, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.stocks = stocks

    def forward(self, inputs):
        # inputs: [stocks, time_steps, channels]
        out, _ = self.lstm(inputs)  # [stocks, time_steps, hidden_dim]
        out = out[:, -1, :]  # [stocks, hidden_dim]
        out = self.fc(out)  # [stocks, 1]
        return out

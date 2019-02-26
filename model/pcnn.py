import torch
import torch.nn as nn


class CNNwithPool(nn.Module):
    def __init__(self, cnn_layers, kernel_size):
        super(CNNwithPool, self).__init__()
        self.cnn = nn.Conv2d(1, cnn_layers, kernel_size)
        self.cnn.bias.data.copy_(nn.init.constant(self.cnn.bias.data, 0.))

    def forward(self, x, mask):

        cnn_out = self.cnn(x).squeeze(3)
        mask = mask[:, :, :cnn_out.size(2)].transpose(0,1)
        pcnn_out, _ = torch.max(cnn_out.unsqueeze(1) + mask.unsqueeze(2), 3)

        return pcnn_out

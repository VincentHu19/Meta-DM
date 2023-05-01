import torch.nn as nn
import torch
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=1600):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, 320),
            conv_block(320, z_dim),
        )
        self.out_channels = 1600
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)


    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def loss(self, data, num_way, num_support, num_query):
        p = num_support * num_way
        data_shot, data_query = data[:p], data[p:]

        proto = self.forward(data_shot)
        proto = proto.reshape(num_support, num_way, -1).mean(dim=0)

        label = torch.arange(num_way).repeat(num_query)
        label = label.type(torch.cuda.LongTensor)

        logits = self.scale * euclidean_metric(self.forward(data_query), proto) / self.out_channels
        # logits = euclidean_metric(self.forward(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        return loss, acc


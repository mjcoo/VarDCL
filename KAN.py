'''This is a sample code for the simulations of the paper:
Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

https://arxiv.org/abs/2405.12832
and also available at:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
We used efficient KAN notation and some part of the code:https://github.com/Blealtan/efficient-kan

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear weights for combining outputs
        #self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features)) #not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)
    def forward(self, x):
        #You may like test the cases like Spl-KAN
        base_output = F.linear(self.base_activation(x), self.weight1)
        # Apply batch normalization
        return self.bn(base_output)
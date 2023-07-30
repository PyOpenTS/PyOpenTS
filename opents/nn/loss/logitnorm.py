import torch.nn as nn
import torch
import torch.nn.functional as F

class LogitNorm(nn.Module):
    def __init__(self, device, t=1.0):
        super(LogitNorm, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, labels):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, labels)
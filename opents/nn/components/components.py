# coding=utf-8
import torch.nn as nn

class Lambda(nn.Module):
    
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

class Residual(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        h = self.model(x)
        return x + h



class InitializedLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

class InitializedConv1d(nn.Conv1d):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

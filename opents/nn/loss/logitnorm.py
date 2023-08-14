import torch.nn as nn
import torch
import torch.nn.functional as F

class LogitNorm(nn.Module):
    """
    LogitNorm Class: A PyTorch module to compute the logit normalization of the input tensor followed by
    the cross entropy loss.

    :param device: torch.device, The device on which the module will run.
    :param t: float, optional, default=1.0, The temperature parameter for scaling the logits.

    Examples:
    ---------
    >>> logit_norm = LogitNorm(device=torch.device('cpu'))
    >>> x = torch.randn(16, 10)
    >>> labels = torch.randint(0, 10, (16,))
    >>> loss = logit_norm(x, labels)
    """
    def __init__(self, device, t=1.0):
        super(LogitNorm, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, labels):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, labels)
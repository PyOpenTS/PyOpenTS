import torch
import torch.nn as nn
from opents.nn.components.components import Lambda, Residual, InitializedLinear, InitializedConv1d

class FCN(nn.Module):
    """
    Fully Connected Network (FCN) for timeseries classification.

    Args:
            in_channels (int): Number of input channels.
            unit_list (list): List of number of units in each convolutional layer.
            out_channels (int): Number of output channels.
            num_classes (int): Number of classes for classification.
            num_cnns (int): Number of convolutional layers.
            kernel_size (int, optional): Size of the kernel for the convolutional layers. Defaults to 30.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.5.
            padding (str, optional): Padding type for the convolutional layers. Can be 'same' or 'valid'. Defaults to 'same'.

    Examples:
    ---------
    >>> fcn = FCN(in_channels=1, unit_list=[64, 128], out_channels=10, num_classes=3, num_cnns=2)
    >>> x = torch.randn(16, 1, 500)
    >>> output = fcn(x)
    """
    def __init__(self, in_channels, unit_list, out_channels, num_classes, num_cnns, kernel_size=30, stride=1, dropout=0.5, padding='same'):
        super(FCN, self).__init__()
        self.in_channels = in_channels
        self.unit_list = unit_list
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_cnns = num_cnns
        self.kernel = kernel_size
        self.stride = stride
        self.dropout = nn.Dropout(dropout)

        padding_val="same" if stride == 1 else self.kernel // 2
        self.convs = nn.ModuleList()

        self.convs.append(Lambda(lambda x: torch.permute(x, [0, 2, 1])))

        # first cnns layer
        self.convs.append(InitializedConv1d(self.in_channels, self.unit_list[0], kernel_size=self.kernel, stride=self.stride, padding=padding_val))
        self.convs.append(nn.BatchNorm1d(self.unit_list[0]))
        self.convs.append(nn.ReLU())
        self.convs.append(self.dropout)

        # other cnns layers: except first cnns layer
        for unit_num in range(len(self.unit_list)):
            for cnns_num in range(self.num_cnns):
                    self.convs.append(Residual(InitializedConv1d(self.unit_list[unit_num], self.unit_list[unit_num], kernel_size=self.kernel, padding=padding_val)))
                    self.convs.append(nn.BatchNorm1d(self.unit_list[unit_num]))
                    self.convs.append(nn.ReLU())
                    self.convs.append(self.dropout)
            if unit_num == len(self.unit_list) - 1:
                 break
            self.convs.append(InitializedConv1d(self.unit_list[unit_num], self.unit_list[unit_num + 1], kernel_size=self.kernel, stride=self.stride, padding=padding_val))
            self.convs.append(nn.BatchNorm1d(self.unit_list[unit_num + 1]))
            self.convs.append(nn.ReLU())
            self.convs.append(self.dropout)

        self.convs.append(Lambda(lambda x: torch.permute(x, [0, 2, 1])))
        self.convs.append(Lambda(lambda x: torch.concat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)))
        self.convs.append(self.dropout) 

        # add the last linear layer
        self.convs.append(InitializedLinear(unit_list[-1] * 2, num_classes))
    
    def forward(self, inputs):
        x = inputs
        for layer in self.convs:
            x = layer(x)
        return x

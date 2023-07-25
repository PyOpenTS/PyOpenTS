import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    Fully Connected Network (FCN) for timeseries classification.
    """
    def __init__(self, in_channels, unit_list, out_channels, num_classes, num_cnns, kernel_size=30, stride=1, dropout=0.5, padding='same'):
        """
        Constructor for the FCN class.

        Args:
            in_channels (int): Number of input channels.
            unit_list (list): List of number of units in each layer.
            out_channels (int): Number of output channels.
            num_classes (int): Number of classes for classification.
            num_cnns (int): Number of convolutional layers.
            kernel_size (int, optional): Size of the kernel for the convolutional layers. Defaults to 30.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            dropout (float, optional): Dropout rate. Defaults to 0.5.
            padding (str, optional): Padding type for the convolutional layers. Defaults to 'same'.
        """
        super(FCN, self).__init__()
        self.in_channels = in_channels
        self.unit_list = unit_list
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_cnns = num_cnns
        self.kernel = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.padding = padding

        self.convs = nn.ModuleList()

        # first cnns layer
        self.convs.append(nn.Conv1d(self.in_channels, self.unit_list[0], kernel_size=self.kernel, stride=self.stride, padding=self.padding if self.stride == 1 else self.kernel // 2))
        self.convs.append(nn.ReLU())
        self.convs.append(nn.Dropout(self.dropout))

        # other cnns layers: except first cnns layer
        for unit_num in range(len(self.unit_list)):
            for cnns_num in range(self.num_cnns):
                    self.convs.append(Residual(nn.Conv1d(self.unit_list[unit_num], self.unit_list[unit_num], kernel_size=self.kernel, padding=self.padding)))
                    self.convs.append(nn.ReLU())
                    self.convs.append(nn.Dropout(self.dropout))
            if unit_num == len(self.unit_list) - 1:
                 break
            self.convs.append(nn.Conv1d(self.unit_list[unit_num], self.unit_list[unit_num + 1], kernel_size=self.kernel, stride=self.stride, padding=self.padding))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.Dropout(self.dropout))
        
        # the last linear layer
        self.fc = nn.Linear(unit_list[-1] * 2, num_classes)
    
    def forward(self, inputs):
        """
        Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = inputs
        x = x.transpose(1, 2)
        
        for layer in self.convs:
            x = layer(x)
        x = x.transpose(1, 2)

        # last layer
        x = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        x = nn.Dropout(0.5)(x)
        x = self.fc(x)

        return x

class Residual(nn.Module):
    """
    Implements the Residual Block for the FCN.
    """
    def __init__(self, model):
        """
        Constructor for the Residual class.

        Args:
            model (nn.Module): The base model/block on which residual connections will be used.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding the input tensor and the output from the base model/block.
        """
        h = self.model(x)
        return x + h
import torch
import torch.nn as nn

class FCN(nn.Module):

    def __init__(self, input_size, unit_list, output_size, num_classes, num_cnns, kernel_size=30, stride=1, dropout=0.5,padding='same'):
        super(FCN, self).__init__()
        self.input_size = input_size
        self.unit_list = unit_list
        self.output_size = output_size
        self.num_classes = num_classes
        self.num_cnns = num_cnns
        self.kernel = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.padding = "same"

        self.convs = nn.ModuleList()

        # first cnns layer
        self.convs.append(nn.Conv1d(self.input_size, self.unit_list[0], kernel_size=self.kernel, stride=self.stride, padding=self.padding if self.stride == 1 else self.kernel // 2))
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
        self.fc = nn.Linear(unit_list[-1], num_classes)
    
    def forward(self, inputs):
        x = inputs
        x = x.transpose(1, 2)
        
        for layer in self.convs:
            x = layer(x)
        x = x.transpose(1, 2)

        # last layer
        torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        x = nn.Dropout(0.5)(x)
        x = self.fc(x)

        return x

class Residual(nn.Module):
    def __init__(self, model):
         super().__init__()
         self.model = model

    def forward(self, x):
        h = self.model(x)
        return x + h
import torch
import torch.nn as nn
from opents.datasets import TSDataset
from opents.utils.data_utils import RandomSplitOpenDataset
from torch.utils.data import TensorDataset, DataLoader
from opents.nn.models import FCN
from tqdm import tqdm

device = torch.device('cuda:9' if torch.cuda.is_available else 'cpu')

# load the dataset from ucr
x_train, y_train, x_test, y_test = TSDataset(dataset_name='SmoothSubspace', dataset_root_path='UCR', datasets_name='ucr').load()

# x_train, y_train, x_test, y_test = RandomSplitOpenDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, train_size_rate=0.5, open_label_rate=0.3).load()

# merge the data and labels
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# generate train dataloader and test dataloader
batch_size = 300
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers= 10, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers= 10, pin_memory=True)

# construct fcn model 
model = FCN(in_channels=x_train.shape[-1], unit_list=[64, 128], out_channels=len(torch.unique(y_train)), num_classes=len(torch.unique(y_train)), num_cnns=3, stride=1, padding="same")
model.to(device)

# loss funciton
criterion = nn.CrossEntropyLoss()

# optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# main loop
num_epochs = 100
ave_loss = 0
test_loss = 0
for epoch in tqdm(range(num_epochs)):

    model.train()
    for batch_inputs, batch_labels in train_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        logit = model(batch_inputs)
        loss = criterion(logit, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ave_loss = 0.8 * ave_loss + 0.2 * loss
    

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in test_dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            logit = model(batch_inputs)
            loss = criterion(logit, batch_labels)

            test_loss = test_loss * 0.8 + loss * 0.2
    best_test_loss = float('inf')
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), f'best_logits.pt')

    print("Epoch:{0:3d}\t|Train_Loss:{1:20f}\t|Test_Loss:{2:6f}".format(epoch + 1, ave_loss, test_loss))
    
# %%
import torch
import torch.nn as nn
from opents.datasets import TSDataset
from opents.utils.data_utils import RandomSplitOpenDataset, relabel_from_zero, preprocess_test_labels
from torch.utils.data import TensorDataset, DataLoader
from opents.nn.models import FCN
from tqdm import tqdm
import torchmetrics
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from opents.nn import LogitNorm
import numpy as np
import argparse

best_temp = 0.0
best_ood_accuracy = 0.0
batch_ave_wrong_probabilities = 0.0

# add parser in the file
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='Crop', help="dataset name")
parser.add_argument("--dataset_root_path", type=str, default='UCR', help="the dataset path, for example:UCR/Crop, you should chose the dataset root path: UCR")
parser.add_argument("--datasets_name", type=str, default='ucr', help="")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--epochs', type=int, default=1000, help="the num of training epochs")
parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


cudnn.benchmark = True

# args
criterion_name = "LogitNorm"

# load the dataset from ucr
x_train, y_train, x_test, y_test = TSDataset(dataset_name=args.dataset_name, dataset_root_path=args.dataset_root_path, datasets_name=args.datasets_name).load()

# merge x_train and x_test, merge y_train and y_test
x_train, y_train, x_test, y_test = x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()
x_all = np.concatenate([x_train, x_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

# change x_all , y_all to tensor
x_all, y_all = torch.tensor(x_all), torch.tensor(y_all)

# random split dataset to x_train, y_train, x_val, y_val, x_test, y_test 
x_train, y_train, x_val, y_val, x_test, y_test = RandomSplitOpenDataset(x_all=x_all, y_all=y_all, train_size_rate=0.6, test_size_rate=0.2, open_label_rate=0.2, train_random_state=42, open_random_state=42).load()

# [1, 3, 5]
real_label = torch.unique(y_train)
# [0, 1, 2]
y_train, y_val = relabel_from_zero(y_train, y_val)
# [0, 1, 2, 3, 4, 5] --->[3, 0, 3, 1, 3, 2]
y_test = preprocess_test_labels(y_test, y_train, real_label)

num_classes = int(y_train.max()) + 1
# merge the data and labels
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

# generate train dataloader and test dataloader
batch_size = 4096
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# for temp in np.arange(0.00001, 0.0005, 0.00001):


# construct fcn model
# 没有batchnorm 
model = FCN(in_channels=x_train.shape[-1], unit_list=[64, 128, 256], out_channels=num_classes, num_classes=num_classes, num_cnns=0, stride=1, padding="same")
model.to(args.device)

# loss funciton
criterion = LogitNorm(args.device, 1e-5) 

# optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

metric_dict = {
    "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    "loss":torchmetrics.MeanMetric()
}

metric_dict = {
    metric_name: metric.to(args.device) for metric_name, metric in metric_dict.items()
}

# main loop
num_epochs = 501
best_accuracy = 0.0
best_cls_loss = float('inf')

for epoch in tqdm(range(args.epochs)):

    model.train()
    for batch_inputs, batch_labels in train_dataloader:
        batch_inputs = batch_inputs.to(args.device)
        batch_labels = batch_labels.to(args.device)

        logit = model(batch_inputs)
        cls_loss = criterion(logit, batch_labels)
        # cls_loss = cls_losses.mean()

        l2_loss = 0.0
        for name, param in model.named_parameters():
            if "weight" in name:
                l2_loss += (param ** 2).sum() * 0.5
            
        loss = cls_loss + l2_loss * 1e-5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:

        model.eval()
        with torch.no_grad():

            metric_dict['accuracy'].reset()
            metric_dict['loss'].reset()

            for batch_inputs, batch_labels in val_dataloader:
                batch_inputs = batch_inputs.to(args.device)
                batch_labels = batch_labels.to(args.device)

                logit = model(batch_inputs)

                preds = logit.argmax(dim=-1)

                test_cls_loss = criterion(logit, batch_labels)
                metric_dict["accuracy"](preds, batch_labels)
                metric_dict["loss"](test_cls_loss)

            accuracy = metric_dict["accuracy"].compute().detach().cpu().numpy()
            test_cls_loss = metric_dict["loss"].compute().detach().cpu().numpy()

        if accuracy >= best_accuracy:
            if accuracy > best_accuracy or test_cls_loss < best_cls_loss:
                best_cls_loss = test_cls_loss
                best_accuracy = accuracy 
                # torch.save(model.state_dict(), f'{args.dataset_name}_LogitNorm_temp_simplemodel.pt')
                # print("save model:{}\n".format(best_accuracy))

        # print("Epoch:{0:3d}\t|train_loss:{1:15f}\t|cls_loss:{2:6f}\t|Accuarcy:{3:6f}\t".format(epoch + 1, cls_loss, test_cls_loss, accuracy))


# %%
# # test ood dataset.
model.eval()
ood_accuracy = 0
original_accuracy = 0
threshold = 0
batch_ave_wrong_probabilities = 0
with torch.no_grad():
    for num, (batch_inputs, batch_labels) in enumerate(test_dataloader):
        batch_inputs = batch_inputs.to(args.device)
        batch_labels = batch_labels.to(args.device)

        logits = model(batch_inputs)

        probabilities = torch.nn.functional.softmax(logits, dim=1)

        prediction_labels = torch.argmax(probabilities, dim=1)
        probabilities_max = torch.max(probabilities, dim=1)[0]

        mask = (prediction_labels != batch_labels)
        batch_ave_wrong_probabilities += probabilities_max[mask].mean()
        
        original_accuracy += (prediction_labels == batch_labels).sum().item()
        ood_accuracy += (prediction_labels == batch_labels).sum().item()
    threshold = batch_ave_wrong_probabilities / (num + 1)
    # print("threadhold:{}".format(threshold))
    # print("original Accuracy:{}".format(original_accuracy / len(y_test)))
    # print("ood Accuracy:{}".format(ood_accuracy / len(y_test)))

ood_accuracy = 0
for num, (batch_inputs, batch_labels) in enumerate(test_dataloader):
    batch_inputs = batch_inputs.to(args.device)
    batch_labels = batch_labels.to(args.device)

    logits = model(batch_inputs)

    probabilities = torch.nn.functional.softmax(logits, dim=1)

    prediction_labels = torch.argmax(probabilities, dim=1)
    probabilities_max = torch.max(probabilities, dim=1)[0]

    prediction_labels = torch.where(probabilities_max < threshold, torch.tensor(torch.unique(y_test)[-1]).to(args.device), prediction_labels)

    # mask = (prediction_labels != batch_labels)
    # batch_ave_wrong_probabilities += probabilities_max[mask].mean()
    
    ood_accuracy += (prediction_labels == batch_labels).sum().item()
ood_accuracy = ood_accuracy / len(y_test) 

print("best accuracy in val dataset:", best_accuracy)
print("threadhold:{}".format(threshold))
print("original Accuracy:{}".format(original_accuracy / len(y_test)))
print("ood Accuracy:{}".format(ood_accuracy))
if ood_accuracy > best_ood_accuracy:
    best_ood_accuracy = ood_accuracy
    # best_temp = temp
print("best_ood_accuracy:", best_ood_accuracy)
# print("best_temp:", best_temp)

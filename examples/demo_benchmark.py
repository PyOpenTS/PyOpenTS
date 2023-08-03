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

device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# load the dataset from ucr
x_train, y_train, x_test, y_test = TSDataset(dataset_name='Crop', dataset_root_path='UCR', datasets_name='ucr').load()

x_train, y_train, x_val, y_val, x_test, y_test = RandomSplitOpenDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, train_size_rate=0.4, test_size_rate=0.2, open_label_rate=0.2, train_random_state=42, open_random_state=42).load()

real_label = torch.unique(y_train)

y_train, y_val = relabel_from_zero(y_train, y_val)

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

# construct fcn model 
model = FCN(in_channels=x_train.shape[-1], unit_list=[64, 128, 256], out_channels=num_classes, num_classes=num_classes, num_cnns=3, stride=1, padding="same")
model.to(device)


# loss funciton
criterion = nn.CrossEntropyLoss()

# optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

metric_dict = {
    "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    "loss":torchmetrics.MeanMetric()
}

metric_dict = {
    metric_name: metric.to(device) for metric_name, metric in metric_dict.items()
}

# main loop
num_epochs = 1000
best_accuracy = 0.0
best_cls_loss = float('inf')

for epoch in tqdm(range(num_epochs)):

    model.train()
    for batch_inputs, batch_labels in train_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        logit = model(batch_inputs)
        cls_losses = criterion(logit, batch_labels)
        cls_loss = cls_losses.mean()

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

            for batch_inputs, batch_labels in val_dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

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
                torch.save(model.state_dict(), f'crop_best_logits_0.001.pt')
                print("save model:{}\n".format(best_accuracy))

        print("Epoch:{0:3d}\t|train_loss:{1:15f}\t|cls_loss:{2:6f}\t|Accuarcy:{3:6f}\t".format(epoch + 1, cls_loss, test_cls_loss, accuracy))


# %%
# # test ood dataset.
model.load_state_dict(torch.load("crop_best_logits_0.001.pt"))

model.eval()
ood_accuracy = 0
original_accuracy = 0
threshold = 0
batch_ave_wrong_probabilities = 0
with torch.no_grad():
    for num, (batch_inputs, batch_labels) in enumerate(test_dataloader):
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_inputs)

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        print(probabilities)
        prediction_labels = torch.argmax(probabilities, dim=1)
        probabilities_max = torch.max(probabilities, dim=1)[0]

        mask = (prediction_labels != batch_labels)
        batch_ave_wrong_probabilities += probabilities_max[mask].mean()
        
        original_accuracy += (prediction_labels == batch_labels).sum().item()
        ood_accuracy += (prediction_labels == batch_labels).sum().item()
    threshold = batch_ave_wrong_probabilities / (num + 1)
    print("threadhold:{}".format(threshold))
    print("original Accuracy:{}".format(original_accuracy / len(y_test)))
    print("ood Accuracy:{}".format(ood_accuracy / len(y_test)))

ood_accuracy = 0
for num, (batch_inputs, batch_labels) in enumerate(test_dataloader):
    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)

    logits = model(batch_inputs)

    probabilities = torch.nn.functional.softmax(logits, dim=1)

    prediction_labels = torch.argmax(probabilities, dim=1)
    probabilities_max = torch.max(probabilities, dim=1)[0]

    prediction_labels = torch.where(probabilities_max < threshold, torch.tensor(torch.unique(y_test)[-1]).to(device), prediction_labels)

    mask = (prediction_labels != batch_labels)
    batch_ave_wrong_probabilities += probabilities_max[mask].mean()
    
    ood_accuracy += (prediction_labels == batch_labels).sum().item()
print("threadhold:{}".format(threshold))
print("original Accuracy:{}".format(original_accuracy / len(y_test)))
print("ood Accuracy:{}".format(ood_accuracy / len(y_test)))
# %%


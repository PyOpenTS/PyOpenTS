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
import numpy as np
from opents.nn.detector import compute_mavs_and_dists, weibull, openmax

device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# model parameters
lr = 0.001

# weibull parameters
weibull_tail_size = 20
weibull_alpha = 3
weibull_threshold = 0.9

# load the dataset from ucr, TSDataset fullfill the origin dataset structure.
x_train, y_train, x_test, y_test = TSDataset(dataset_name='Crop', dataset_root_path='UCR', datasets_name='ucr').load()

# make the trainset and testset openly, random split the dataset and choose our dataest structure.
x_train, y_train, x_val, y_val, x_test, y_test = RandomSplitOpenDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, train_size_rate=0.6, test_size_rate=0.2, open_label_rate=0.3, train_random_state=42, open_random_state=42).load()

# y_train and y_val labels.
real_label = torch.unique(y_train)

# relabel the y_train and y_val from zero.
y_train, y_val = relabel_from_zero(y_train, y_val)

y_test = preprocess_test_labels(y_test, y_train, real_label)

# torch.unique(y_test) = torch.unique(y_train) + 1
# delete the one label and data from y_test randomly
x_test, y_test = x_test.numpy(), y_test.numpy()

test_unique_labels = np.unique(y_test)
test_unique_labels_to_remove = y_test[:-1]

random_label_to_remove = np.random.choice(test_unique_labels_to_remove)
index_to_remove = np.where(y_test == random_label_to_remove)[0]

x_test = np.delete(x_test, index_to_remove, axis=0)
y_test = np.delete(y_test, index_to_remove)

# change the numpy() to tensor()
x_test, y_test = torch.tensor(x_test), torch.tensor(y_test)

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
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# accuracy and loss metrics
metric_dict = {
    "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    "loss":torchmetrics.MeanMetric()
}

metric_dict = {
    metric_name: metric.to(device) for metric_name, metric in metric_dict.items()
}

# main loop
num_epochs = 5000
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

# model evaluation
    if epoch % 100 == 0 and epoch != 0:

        scores, labels = [], []
        model.eval()
        with torch.no_grad():

            for batch_inputs, batch_labels in test_dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_inputs)

                scores.append(logit)
                labels.append(batch_labels)

        scores = torch.cat(scores, dim=0).detach().cpu().numpy()
        labels = torch.cat(labels, dim=0).detach().cpu().numpy()

        scores = np.array(scores)[:, np.newaxis, :]
        labels = np.array(labels)

        # fit the weibull distribution
        # mavs: mean activation vectors; dist: distribution
        mavs, dists = compute_mavs_and_dists(num_classes=num_classes, train_dataloader=train_dataloader, device=device, model=model)
        categories = list(range(0, num_classes))

        weibull_model = weibull(means=mavs, dists=dists, categories=categories, tailsize=weibull_tail_size, distance_type='euclidean')

        all_mcv_filled = True
        for mcv in mavs:
            if len(mcv) == 0:
                all_mcv_filled = False
        if all_mcv_filled:
            pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
            score_softmax, score_openmax = [], []
            for score in scores:
                so, ss = openmax(weibull_model, categories, score, 0.5, weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
                pred_softmax.append(np.argmax(ss))
                pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= weibull_threshold else num_classes)
                pred_openmax.append(np.argmax(so) if np.max(so) >= weibull_threshold else num_classes)
                score_softmax.append(ss)
                score_openmax.append(so)
            print("openmax prediction:\n", pred_openmax)
            print("labels:\n", labels)

# # %%
# # # test ood dataset.
# model.load_state_dict(torch.load("crop_best_logits_0.001.pt"))

# model.eval()
# ood_accuracy = 0
# original_accuracy = 0
# threshold = 0
# batch_ave_wrong_probabilities = 0
# with torch.no_grad():
#     for num, (batch_inputs, batch_labels) in enumerate(test_dataloader):
#         batch_inputs = batch_inputs.to(device)
#         batch_labels = batch_labels.to(device)

#         logits = model(batch_inputs)

#         probabilities = torch.nn.functional.softmax(logits, dim=1)
#         print(probabilities)
#         prediction_labels = torch.argmax(probabilities, dim=1)
#         probabilities_max = torch.max(probabilities, dim=1)[0]

#         mask = (prediction_labels != batch_labels)
#         batch_ave_wrong_probabilities += probabilities_max[mask].mean()
        
#         original_accuracy += (prediction_labels == batch_labels).sum().item()
#         ood_accuracy += (prediction_labels == batch_labels).sum().item()
#     threshold = batch_ave_wrong_probabilities / (num + 1)
#     print("threadhold:{}".format(threshold))
#     print("original Accuracy:{}".format(original_accuracy / len(y_test)))
#     print("ood Accuracy:{}".format(ood_accuracy / len(y_test)))

# ood_accuracy = 0
# for num, (batch_inputs, batch_labels) in enumerate(test_dataloader):
#     batch_inputs = batch_inputs.to(device)
#     batch_labels = batch_labels.to(device)

#     logits = model(batch_inputs)

#     probabilities = torch.nn.functional.softmax(logits, dim=1)

#     prediction_labels = torch.argmax(probabilities, dim=1)
#     probabilities_max = torch.max(probabilities, dim=1)[0]

#     prediction_labels = torch.where(probabilities_max < threshold, torch.tensor(torch.unique(y_test)[-1]).to(device), prediction_labels)

#     mask = (prediction_labels != batch_labels)
#     batch_ave_wrong_probabilities += probabilities_max[mask].mean()
    
#     ood_accuracy += (prediction_labels == batch_labels).sum().item()
# print("threadhold:{}".format(threshold))
# print("original Accuracy:{}".format(original_accuracy / len(y_test)))
# print("ood Accuracy:{}".format(ood_accuracy / len(y_test)))
# # %%
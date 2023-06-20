#coding=utf-8

from opents.data import generate_datasets
from opents.datasets import ucr


# load ucr and generate dataloader
ucr_dataset = ucr.UCRDataset("Chinatown",'UCR')
train, train_label, test, test_label = ucr_dataset.load_ucr()

# merge ucr dataset and random choose the train and test data
ucr_open_dataset = generate_datasets.GenerateNewUCR("Chinatown",'UCR')
ucr_open_dataset.merge_ucr()
ucr_open_dataset.label_num()



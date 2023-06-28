#coding=utf-8

from opents.data import datasets


# load ucr and generate dataloader
ucr_dataset = datasets.UCRDataset(dataset_name="Chinatown",dataset_root_path='UCR')
train, train_label, test, test_label = ucr_dataset.load()




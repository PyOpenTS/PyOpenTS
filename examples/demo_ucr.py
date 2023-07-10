#coding=utf-8

from opents.datasets import UCRDataset, UEADataset
import opents
# load ucr and generate dataloader
ucr_dataset = opents.datasets.UCRDataset(dataset_name="Chinatown",dataset_root_path='UCR')
x_train, y_train, x_test, y_test = ucr_dataset.load()

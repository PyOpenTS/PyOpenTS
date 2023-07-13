#coding=utf-8

from opents.datasets import OpenUCRDataset

x_train, y_train, x_test, y_test = OpenUCRDataset(dataset_name='CBF', dataset_root_path='UCR', train_percentage=0.5, open_percentage=0.5)

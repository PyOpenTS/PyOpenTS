#coding=utf-8

from opents.datasets import OpenUCRDataset

x_all, y_all = OpenUCRDataset(dataset_name='Chinatown', dataset_root_path='UCR')
print(x_all)

#coding=utf-8

from opents.datasets import OpenUCRDataset

openuea_dataset = OpenUCRDataset(dataset_name='Chinatown', dataset_root_path='UCR')
x_all, y_all = openuea_dataset.load()
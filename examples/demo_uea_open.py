#coding=utf-8

from opents.datasets import OpenUEADataset

openuea_dataset = OpenUEADataset(dataset_name='LSST', dataset_root_path='UEA')
x_all, y_all = openuea_dataset.load()
print(x_all)
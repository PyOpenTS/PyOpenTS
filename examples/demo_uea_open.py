#coding=utf-8

from opents.datasets import OpenUEADataset

x_train, y_train, x_test, y_test = OpenUEADataset(dataset_name='LSST', dataset_root_path='UEA', train_percentage=0.5, open_percentage=0.5)
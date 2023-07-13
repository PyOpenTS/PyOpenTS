#coding=utf-8

from opents.datasets import UEADataset

# load uea and generate dataloader
x_train, y_train, x_test, y_test = UEADataset(dataset_name="LSST", dataset_root_path='UEA')
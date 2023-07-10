#coding=utf-8

from opents.datasets import UCRDataset, UEADataset
import opents

# load uea and generate dataloader
uea_dataset = UEADataset(dataset_name="LSST", dataset_root_path='UEA')
x_train, y_train, x_test, y_test = uea_dataset.load()

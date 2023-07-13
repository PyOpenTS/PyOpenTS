#coding=utf-8

import opents
# load ucr and generate dataloader
x_train, y_train, x_test, y_test = opents.datasets.UCRDataset(dataset_name="Chinatown",dataset_root_path='UCR')
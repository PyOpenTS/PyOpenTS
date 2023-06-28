#coding=utf-8

from opents.data import datasets


# load ucr and generate dataloader
ucr_dataset = datasets.UCRDataset(dataset_name="Chinatown",dataset_root_path='UCR',datasets_name='UCR')
train, train_label, test, test_label = ucr_dataset.load()

# load uea and generate dataloader
uea_dataset = datasets.UEADataset(dataset_name="LSST", dataset_root_path='UEA',datasets_name='UEA')
train, train_label, test, test_label = uea_dataset.load()
print(train)

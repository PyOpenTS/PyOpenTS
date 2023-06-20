import os
import json
import random
import pandas as pd
import argparse
import torch


class GenerateNewUCR(object):
    
    def __init__(self, dataset_name, ucr_root_path):
        """

        :param dataset_name: 
        :param dataset_root_path:
        """
        self.ucr_root_path = ucr_root_path
        self.dataset_name = dataset_name

    def merge_ucr(self):
        """ merge_ucr is aim to merge all ucr train dataset and test dataset.
        
        Args:
            UCR_path (_type_): UCR folder path
            dataset (_type_): UCR's dataset name
        Returns:
            None
        """

        train_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_TRAIN.tsv")
        test_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_TEST.tsv")
        
        train_df = pd.read_csv(train_path, sep='\t', header=None)
        test_df = pd.read_csv(test_path, sep='\t', header=None)
        
        all_data_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_ALL.tsv")
        all_data = pd.concat([train_df, test_df], axis=0)
        all_data.to_csv(all_data_path, index=False, sep='\t', header=None)
        print("Merging ucr train and test data for dataset {} has been completed" .format(self.dataset_name))

    def label_num(self):
        all_data_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_ALL.tsv")
        all_data = pd.read_csv(all_data_path, sep="\t", header=None)
        all_data_tensor = torch.tensor(all_data.values)
        all_data_label = all_data_tensor[:, 0]
        label_dict = {}
        for i, l in enumerate(all_data_label):
            if label_dict.get(l.item()) is None:
                label_dict[l.item()] = 1
            else:
                label_dict[l.item()] += 1
        return label_dict
    
    def generate_open_json(self):
        datasets = os.listdir(self.base_path)
        result_dict = {}
        for dataset in datasets:
            result_dict[dataset] = self.label_num(dataset)
        with open('open_ucr.json', 'w') as fp:
            json.dump(result_dict, fp)

    def random_train_and_test_dataset(self):
        # load the JSON data into Python
        with open("open_ucr.json", "r") as fin:
            data = json.load(fin)

        # Iterate over each entry in the JSON dictionary.
        for key in data:
            items = list(data[key].items())
            # Randomly shuffle the key-value pairs.
            random.shuffle(items)
            # Divide into three piles.
            n = len(items) // 3
            test_sets = [{k: v for k, v in items[i * n:(i + 1) * n]} for i in range(3)]
            train_sets = [{k: v for k, v in items if (k, v) not in test_set} for test_set in test_sets]
            
            # Generate three test sets and training sets for each entry 
            for i in range(3):
                # Write to json file
                with open(f'{key}_test_set_{i}.json', 'w') as f:
                    json.dump(test_sets[i], f)
                with open(f'{key}_train_set_{i}.json', 'w') as f:
                    json.dump(train_sets[i], f)

class GenerateNewUEA(object):
    
    def __init__(self, dataset_name, ucr_root_path):
        """

        :param dataset_name: 
        :param dataset_root_path:
        """
        self.ucr_root_path = ucr_root_path
        self.dataset_name = dataset_name

    def merge_ucr(self):
        """ merge_ucr is aim to merge all ucr train dataset and test dataset.
        
        Args:
            UCR_path (_type_): UCR folder path
            dataset (_type_): UCR's dataset name
        Returns:
            None
        """

        train_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_TRAIN.tsv")
        test_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_TEST.tsv")
        
        train_df = pd.read_csv(train_path, sep='\t', header=None)
        test_df = pd.read_csv(test_path, sep='\t', header=None)
        
        all_data_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_ALL.tsv")
        all_data = pd.concat([train_df, test_df], axis=0)
        all_data.to_csv(all_data_path, index=False, sep='\t', header=None)
        print("Merging ucr train and test data for dataset {} has been completed" .format(self.dataset_name))

    def label_num(self):
        all_data_path = os.path.join(self.ucr_root_path, self.dataset_name, self.dataset_name + "_ALL.tsv")
        all_data = pd.read_csv(all_data_path, sep="\t", header=None)
        all_data_tensor = torch.tensor(all_data.values)
        all_data_label = all_data_tensor[:, 0]
        label_dict = {}
        for i, l in enumerate(all_data_label):
            if label_dict.get(l.item()) is None:
                label_dict[l.item()] = 1
            else:
                label_dict[l.item()] += 1
        return label_dict
    
    def generate_open_json(self):
        datasets = os.listdir(self.base_path)
        result_dict = {}
        for dataset in datasets:
            result_dict[dataset] = self.label_num(dataset)
        with open('open_ucr.json', 'w') as fp:
            json.dump(result_dict, fp)

    def random_train_and_test_dataset(self):
        # load the JSON data into Python
        with open("open_ucr.json", "r") as fin:
            data = json.load(fin)

        # Iterate over each entry in the JSON dictionary.
        for key in data:
            items = list(data[key].items())
            # Randomly shuffle the key-value pairs.
            random.shuffle(items)
            # Divide into three piles.
            n = len(items) // 3
            test_sets = [{k: v for k, v in items[i * n:(i + 1) * n]} for i in range(3)]
            train_sets = [{k: v for k, v in items if (k, v) not in test_set} for test_set in test_sets]
            
            # Generate three test sets and training sets for each entry 
            for i in range(3):
                # Write to json file
                with open(f'{key}_test_set_{i}.json', 'w') as f:
                    json.dump(test_sets[i], f)
                with open(f'{key}_train_set_{i}.json', 'w') as f:
                    json.dump(train_sets[i], f)

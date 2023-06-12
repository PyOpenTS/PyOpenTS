import json
import random
import os
import pandas as pd
import argparse
import torch

class new_ucr():
    
    def __init__(self, UCR_path, dataset_name):
        self.UCR_path = UCR_path
        self.dataset_name = dataset_name

    def merge_ucr(self):
        """ merge_ucr is aim to merge all ucr train dataset and test dataset.
        
        Args:
            UCR_path (_type_): UCR folder path
            dataset (_type_): UCR's dataset name
        Returns:
            None
        """

        train_path = os.path.join(self.UCR_path, self.dataset_name, self.dataset_name + "_TRAIN.tsv")
        test_path = os.path.join(self.UCR_path, self.dataset_name, self.dataset_name + "_TEST.tsv")
        
        train_df = pd.read_csv(train_path, sep='\t', header=None)
        test_df = pd.read_csv(test_path, sep='\t', header=None)
        
        all_data_path = os.path.join("../", "UCR", self.dataset_name, self.dataset_name + "_ALL.tsv")
        all_data = pd.concat([train_df, test_df], axis=0)
        all_data.to_csv(all_data_path, index=False, sep='\t', header=None)
        print("Merging ucr train and test data for dataset {} has been completed" .format(self.dataset_name))

    def label_num(self):
        all_data_path = os.path.join("../", "UCR", self.dataset_name, "all_" + self.dataset_name + ".tsv")
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
    
    def generate_json(self):
        datasets = os.listdir(self.base_path)
        result_dict = {}
        for dataset in datasets:
            result_dict[dataset] = self.label_num(dataset)
        with open('open_ucr.json', 'w') as fp:
            json.dump(result_dict, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("UCR_path", type=str, help="Please provide the correct absolute path for the UCR data folder. (We recommend using the absolute path for accuracy)")
    parser.add_argument("dataset_file", type=str, help="Please provide the correct UCR dataset file txt.(it is has been provided in our program)")
    args = parser.parse_args()

    # merge the train dataset and test dataset one by one.
    counter = new_ucr(args.UCR_path, '')
    with open(args.dataset_file, 'r') as f:
        datasets = f.read().splitlines()
    for dataset in datasets:
        counter.dataset_name = dataset
        counter.merge_ucr()

    # # read the all open ucr dataset name
    # with open('open_ucr.json', 'r') as f:
    #     data = json.load(f)

    # # 创建空字典，用于存放训练集和测试集
    # train_sets = {}
    # test_sets = {}

    # # 遍历原始数据的每一个键值对
    # for key, value in data.items():
    #     items = list(value.items())
    #     # 随机打乱items
    #     random.shuffle(items)

    #     # 把items分成三份
    #     n = len(items) // 3
    #     test_set = {k: v for k, v in items[:n]}
    #     train_set = {k: v for k, v in items[n:]}

    #     # 把划分好的训练集和测试集存入对应的字典
    #     train_sets[key] = train_set
    #     test_sets[key] = test_set

    # # 把训练集和测试集写入json文件
    # with open('train_sets.json', 'w') as f:
    #     json.dump(train_sets, f)
    # with open('test_sets.json', 'w') as f:
    #     json.dump(test_sets, f)

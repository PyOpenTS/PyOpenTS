# coding=utf-8
import os

import pandas as pd
import torch
import numpy as np
import requests
import zipfile 

DEFAULT_DATASETS_ROOT = "data"

def download_url(url, datasets_name, datasets_root_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(datasets_root_path, 'wb') as file:
            file.write(response.content)
        print(datasets_name + "is successfully downloaded")
    else:
        print(datasets_name + "is not successfully downloaded") 



def get_dataset_root_path(dataset_root_path=None, dataset_name=None, datasets_name=None,
                          datasets_root_path=DEFAULT_DATASETS_ROOT):
    if dataset_root_path is None:
        dataset_root_path = os.path.join(datasets_root_path, dataset_name)
    dataset_root_path = os.path.abspath(dataset_root_path)
    
    ucr_url = 'https://www.timeseriesclassification.com/ClassificationDownloads/Archives/Univariate2018_ts.zip'
    ucr_password = "someone"

    uea_url = "https://www.timeseriesclassification.com/ClassificationDownloads/Archives/Multivariate2018_ts.zip"

    if os.path.exists(dataset_root_path) is False:
        os.makedirs(dataset_root_path, exist_ok=True)
        if datasets_name == "UCR":
            download_url(ucr_url, "UCR", datasets_root_path)

            with zipfile.ZipFile(datasets_root_path + "UCR_TS_Archive_2018.zip", 'r') as zf:
                if zf.setpassword(bytes(ucr_password, 'utf-8')):
                    zf.extractall()
                    print("The ZIP file has been successfully extracted. UCR 2018 zip is successfully unzip.")

        if datasets_name == "UEA":
            download_url(uea_url, "UEA", datasets_root_path)

            with zipfile.ZipFile(datasets_root_path + "Multivariate2018_ts.zip", 'r') as zf:
                zf.extractall()
                print("The ZIP file has been successfully extracted. UEA 2018 zip is successfully unzip.")
    else:
        print("unzip password is worng, please change the unzip password")
    return dataset_root_path


class Dataset(object):
    def __init__(
            self,
            dataset_name,
            dataset_root_path=None,
            datasets_name=None,
            split=False
            ):
        self.dataset_name = dataset_name
        self.dataset_root_path = get_dataset_root_path(dataset_root_path, dataset_name, datasets_name)
        self.split = split

    
    def ts_dataset(self):
        """ The path to read the UCR file, including the path of the training file and the path of the testing file.
        Args:
            dataset(string):Define a variable 'dataset_name' for the name of each file, which can be used to locate the training and testing file paths for each function.
        Returns:
            Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        """
        # Define the paths for the training and testing sets.
        train_file_path = os.path.join(self.dataset_root_path, self.dataset_name, self.dataset_name + "_TRAIN.tsv")
        test_file_path = os.path.join(self.dataset_root_path, self.dataset_name, self.dataset_name + "_TEST.tsv")

        # Read the data from each file in the training set and testing set using the pandas library, which is a library for data processing.
        train_file_df = pd.read_csv(train_file_path, sep='\t', header=None)
        test_file_df = pd.read_csv(test_file_path, sep='\t', header=None)

        # Create a two-dimensional tensor file from the data in the training set and testing set files.
        train_tensor = torch.tensor(train_file_df.values)
        test_tensor = torch.tensor(test_file_df.values)

        # Here, we extract the distinct labels from each dataset.
        train_labels = torch.unique(train_tensor[:, 0])
        test_labels = torch.unique(test_tensor[:, 0])
        # Store the labels in a dictionary, where the key is the original label and the value is the count of that label.
        transform = {}

        for i, l in enumerate(train_labels):
            transform[l.item()] = i

        # Convert the data into a list.
        train_array = train_tensor.tolist()
        train_array = np.array(train_array)
        test_array = test_tensor.tolist()
        test_array = np.array(test_array)

        # Retrieve the time-series data, excluding the labels. 
        train = train_array[:, 1:].astype(np.float64)
        test = test_array[:, 1:].astype(np.float64)

        # Retrieve the values of all new labels and store them in 'train_label'.
        train_label = np.vectorize(transform.get)(train_array[:, 0])

        # Retrieve all label values from the test set and store them in 'test_label'.
        test_label = np.vectorize(transform.get)(test_array[:, 0])

        # Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        return train[..., np.newaxis], train_label, test[..., np.newaxis], test_label

    def opents_dataset(self):
        """ The path to read the UCR file, including the path of the training file and the path of the testing file.
        Args:
            dataset(string):Define a variable 'dataset_name' for the name of each file, which can be used to locate the training and testing file paths for each function.
        Returns:
            Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        """
        # Define the paths for the training and testing sets.
        train_file_path = os.path.join(self.dataset_root_path, self.dataset_name, self.dataset_name + "_TRAIN.tsv")
        test_file_path = os.path.join(self.dataset_root_path, self.dataset_name, self.dataset_name + "_TEST.tsv")

        # Read the data from each file in the training set and testing set using the pandas library, which is a library for data processing.
        train_file_df = pd.read_csv(train_file_path, sep='\t', header=None)
        test_file_df = pd.read_csv(test_file_path, sep='\t', header=None)
        
        # merge the train and test file into a full data
        all_data_df = pd.concat([train_file_df, test_file_df], axis=0)
        
        # Create a two-dimensional tensor file from the data in the all files.
        all_tensor = torch.tensor(all_data_df.values)
        # train_array = np.array(train_file_df)

        # Here, we extract the distinct labels from each dataset.
        all_labels = torch.unique(all_tensor[:, 0])
        
        # Store the labels in a dictionary, where the key is the original label and the value is the count of that label.
        transform = {}

        for i, l in enumerate(all_labels):
            transform[l.item()] = i

        # Convert the data into a list.
        all_array = all_tensor.tolist()
        all_array = np.array(all_array)

        # Retrieve the time-series data, excluding the labels. 
        all = all_array[:, 1:].astype(np.float64)

        # Retrieve the values of all new labels and store them in 'train_label'.
        all_label = np.vectorize(transform.get)(all_array[:, 0])

        # Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        return all[..., np.newaxis], all_label
    

class TSDataset(Dataset):
    def __init__(self, dataset_name, dataset_root_path=None, split=True):
        super().__init__(dataset_name, dataset_root_path, split)

    def load(self):
        # Call the parent class's ucr_dataset method to get the data
        train_data, train_labels, test_data, test_labels = self.ts_dataset()

        # Modify or add any additional processing specific to TSDataset here

        return train_data, train_labels, test_data, test_labels
    
    
class UCRDataset(TSDataset):
    def __init__(self, dataset_name, dataset_root_path=None, split=True):
        super().__init__(dataset_name, dataset_root_path, split)

class UEADataset(TSDataset):
    def __init__(self, dataset_name, dataset_root_path=None, split=True):
        super().__init__(dataset_name, dataset_root_path, split)

class OpenDataset(Dataset):
    def __init__(self, dataset_name, dataset_root_path=None, split=False):
        super().__init__(dataset_name, dataset_root_path, split)

    def load(self):
        # Call the parent class's open_ucr_dataset method to get the data
        all_data, all_labels = self.opents_dataset()

        # Modify or add any additional processing specific to OpenDataset here

        return all_data, all_labels
    
class OpenUCRDataset(OpenDataset):
    def __init__(self, dataset_name, dataset_root_path=None, split=False):
        super().__init__(dataset_name, dataset_root_path, split)

class OpenUEADataset(OpenDataset):
    def __init__(self, dataset_name, dataset_root_path=None, split=False):
        super().__init__(dataset_name, dataset_root_path, split)

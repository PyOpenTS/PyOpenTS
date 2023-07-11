# coding=utf-8
import os

import pandas as pd
import torch
import numpy as np
from scipy.io import arff
from opents.utils.data_utils import data_file_type, get_dataset_root_path
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(
            self,
            dataset_name,
            dataset_root_path=None,
            datasets_root_name=None,
            split=False
            ):
        """_summary_

        Args:
            dataset_name (_type_): _description_
            dataset_root_path (_type_, optional): _description_. Defaults to None.
            datasets_root_name (_type_, optional): _description_. Defaults to None.
            split (bool, optional): _description_. Defaults to False.
        """
        self.dataset_name = dataset_name
        self.dataset_root_path = get_dataset_root_path(dataset_root_path, dataset_name, datasets_root_name)
        self.datasets_root_name = datasets_root_name
        self.split = split

    
    def ts_dataset(self):
        """ The path to read the UCR or UEA file, including the path of the training file and the path of the testing file.
        Args:
            dataset(string):Define a variable 'dataset_name' for the name of each file, which can be used to locate the training and testing file paths for each function.
        Returns:
            for UCR:add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
            for UEA:return the train data, new label of train data, test data, and new label of test data.
        """
        train_file_df, test_file_df = data_file_type(self.dataset_name, self.dataset_root_path, self.datasets_root_name)
        
        if self.datasets_root_name.lower() == 'ucr':
            # Create a two-dimensional tensor file from the data in the training set and testing set files.
            train_tensor = torch.tensor(train_file_df.values)
            test_tensor = torch.tensor(test_file_df.values)

            # Here, we extract the distinct labels from each dataset.
            train_labels = torch.unique(train_tensor[:, 0])
            test_labels = torch.unique       
            
            # Convert the data into a list.
            train_array = train_tensor.tolist()
            train_array = np.array(train_array)
            test_array = test_tensor.tolist()
            test_array = np.array(test_array)

            # Retrieve the time-series data, excluding the labels. 
            x_train = train_array[:, 1:].astype(np.float64)
            x_test = test_array[:, 1:].astype(np.float64)
            # Store the labels in a dictionary, where the key is the original label and the value is the count of that label.
            transform = {}

            for i, l in enumerate(train_labels):
                transform[l.item()] = i
            # Retrieve the values of all new labels and store them in 'train_label'.
            y_train = np.vectorize(transform.get)(train_array[:, 0])

            # Retrieve all label values from the test set and store them in 'test_label'.
            y_test = np.vectorize(transform.get)(test_array[:, 0])
            # if dataset's dimention is 2, we change the dimention to 3
            x_train = x_train[..., np.newaxis]
            x_test = x_test[..., np.newaxis]           

        elif self.datasets_root_name.lower() == 'uea':
            train_dataset_name = self.dataset_name + "_TRAIN"
            test_dataset_name = self.dataset_name + "_TEST"

            # Store the train data and test data in list, and extend its dimention to 3.
            train_array = np.array(train_file_df[train_dataset_name].tolist())
            x_train = train_array.view((np.float64, len(train_array.dtype.names)))

            test_array = np.array(test_file_df[train_dataset_name].tolist())
            x_test = test_array.view((np.float64, len(test_array.dtype.names)))

            # Here, we extract the distinct labels from each dataset.
            train_labels = torch.unique(torch.tensor(pd.to_numeric(train_file_df['target'], errors='coerce')))
            test_labels = torch.unique(torch.tensor(pd.to_numeric(test_file_df['target'], errors='coerce')))
            transform = {}

            for i, l in enumerate(train_labels):
                transform[l.item()] = i
            
            # Retrieve the values of all new labels and store them in 'train_label'.
            y_train = np.vectorize(transform.get)(train_labels)

            # Retrieve all label values from the test set and store them in 'test_label'.
            y_test = np.vectorize(transform.get)(test_labels) 


        # Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        return x_train, y_train, x_test, y_test

    def opents_dataset(self):
        """ The path to read the UCR or UEA file, including the path of the training file and the path of the testing file.
        Args:
            dataset(string):Define a variable 'dataset_name' for the name of each file, which can be used to locate the training and testing file paths for each function.
        Returns:
            for UCR:add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
            for UEA:return the train data, new label of train data, test data, and new label of test data.
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
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name=None, split=True):
        super().__init__(dataset_name, dataset_root_path, datasets_name, split)

    def load(self):
        # Call the parent class's ucr_dataset method to get the data
        train_data, train_labels, test_data, test_labels = self.ts_dataset()

        # Modify or add any additional processing specific to TSDataset here

        return train_data, train_labels, test_data, test_labels
    
    
class UCRDataset(TSDataset):
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name='UCR', split=True):
        super().__init__(dataset_name, dataset_root_path, datasets_name, split)
        self.datasets_name = datasets_name

class UEADataset(TSDataset):
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name='UEA', split=True):
        super().__init__(dataset_name, dataset_root_path, datasets_name, split)
        self.datasets_name = datasets_name



class OpenDataset(Dataset):
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name=None, split=False):
        super().__init__(dataset_name, dataset_root_path, datasets_name, split)

    def load(self):
        # Call the parent class's open_ucr_dataset method to get the data
        all_data, all_labels = self.opents_dataset()

        # Modify or add any additional processing specific to OpenDataset here

        return all_data, all_labels
    
class OpenUCRDataset(OpenDataset):
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name='UCR', split=False):
        super().__init__(dataset_name, dataset_root_path, datasets_name, split)
        self.datasets_name = datasets_name

class OpenUEADataset(OpenDataset):
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name='UEA', split=False):
        super().__init__(dataset_name, dataset_root_path, datasets_name, split)
        self.datasets_name = datasets_name

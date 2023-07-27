# coding=utf-8

import pandas as pd
import torch
import numpy as np
from opents.utils.data_utils import data_file_type, get_dataset_root_path
from torch import cat

class Dataset:
    def __init__(
            self,
            dataset_name,
            dataset_root_path=None,
            datasets_root_name=None,
            split=False,
            train_percentage=None,
            open_percentage=None,
            random_state=None
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
        self.train_percentage = train_percentage
        self.open_percentage = open_percentage
        self.random_state = random_state
    
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
            test_labels = torch.unique(test_tensor[:, 0]) 
            
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
            test_dataset_name = self.dataset_name + "_TRAIN"

            # Store the train data and test data in list, and extend its dimention to 3.
            train_array = np.array(train_file_df[train_dataset_name].tolist())
            x_train = train_array.view((np.float64, len(train_array.dtype.names)))

            test_array = np.array(test_file_df[test_dataset_name].tolist())
            x_test = test_array.view((np.float64, len(test_array.dtype.names)))

            # Here, we extract the distinct labels from each dataset.
            train_labels = torch.tensor(pd.to_numeric(train_file_df['target'], errors='coerce'))
            train_label = torch.unique(train_labels)
            test_labels = torch.tensor(pd.to_numeric(test_file_df['target'], errors='coerce'))
            test_label = torch.unique(test_labels)
            
            transform = {}

            for i, l in enumerate(train_label):
                transform[l.item()] = i
            
            # Retrieve the values of all new labels and store them in 'train_label'.
            y_train = np.vectorize(transform.get)(train_labels)

            # Retrieve all label values from the test set and store them in 'test_label'.
            y_test = np.vectorize(transform.get)(test_labels) 

        x_train, y_train, x_test, y_test = torch.tensor(x_train).to(torch.float), torch.tensor(y_train).long(), torch.tensor(x_test).to(torch.float), torch.tensor(y_test).long()
        # x_train, y_train, x_test, y_test = x_train.to(torch.float32), y_train.to(torch.long), x_test.to(torch.float32), y_test.to(torch.long)
        # Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        return x_train, y_train, x_test, y_test


    

class TSDataset(Dataset):
    """
    This class is a child of the Dataset class. It aims to handle time-series data.

    Attributes:
    ------------
    dataset_name (str): Name of the dataset
    dataset_root_path (str, optional): Root path of the dataset. Default is None.
    datasets_name (str, optional): Name of the Root datasets. eg:ucr or uea. Default is None.
    x_train (numpy.ndarray): The training data.
    y_train (numpy.ndarray): The labels of the training data.
    x_test (numpy.ndarray): The testing data.
    y_test (numpy.ndarray): The labels of the testing data.

    Methods:
    ------------
    __init__(self, dataset_name, dataset_root_path=None, datasets_name=None):
        Initializes the TSDataset object with the given parameters and loads the dataset using ts_dataset() function.
    
    load(self):
        Returns the loaded dataset including x_train, y_train, x_test, y_test.
    """
    def __init__(self, dataset_name, dataset_root_path=None, datasets_name=None):
        super().__init__(dataset_name, dataset_root_path, datasets_name)
        self.datasets_name = datasets_name
        self.x_train, self.y_train, self.x_test, self.y_test = self.ts_dataset()
    
    def load(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

        

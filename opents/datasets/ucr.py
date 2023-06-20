# coding=utf-8
import os

import pandas as pd
import torch
import numpy as np


class UCRDataset(object):

    def __init__(self, dataset_name, dataset_root_path=None):
        """

        :param dataset_name: 
        :param dataset_root_path:
        """
        self.dataset_name = dataset_name
        self.dataset_root_name = dataset_root_path 

    # UCR dataset url : https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    def load_ucr(self):
        """ The path to read the UCR file, including the path of the training file and the path of the testing file.
        Args:
            dataset(string):Define a variable 'dataset_name' for the name of each file, which can be used to locate the training and testing file paths for each function.
        Returns:
            Add an extra dimension to the train data and test data, then return the train data, new label of train data, test data, and new label of test data.
        """
        # Define the paths for the training and testing sets.
        train_file_path = os.path.join(self.dataset_root_name, self.dataset_name, self.dataset_name + "_TRAIN.tsv")
        test_file_path = os.path.join(self.dataset_root_name, self.dataset_name, self.dataset_name + "_TEST.tsv")

        # Read the data from each file in the training set and testing set using the pandas library, which is a library for data processing.
        train_file_df = pd.read_csv(train_file_path, sep='\t', header=None)
        test_file_df = pd.read_csv(test_file_path, sep='\t', header=None)

        # Create a two-dimensional tensor file from the data in the training set and testing set files.
        train_tensor = torch.tensor(train_file_df.values)
        test_tensor = torch.tensor(test_file_df.values)
        # train_array = np.array(train_file_df)

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
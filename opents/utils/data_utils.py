import os
import requests
import zipfile 
import pandas as pd
from scipy.io import arff
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder
import math

DEFAULT_DATASETS_ROOT = "data"

def download_file(url, datasets_name, datasets_root_path):
    """
    Downloads a file from the specified URL to a destination path.

    Args:
        url (str): URL of the file to download.
        datasets_name (str): Name of the dataset being downloaded.
        datasets_root_path (str): The destination path where the downloaded file will be stored.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(datasets_root_path, 'wb') as file:
            file.write(response.content)
        print(datasets_name + "is successfully downloaded")
    else:
        print(datasets_name + "is not successfully downloaded")


def get_dataset_root_path(dataset_root_path=None, dataset_name=None, datasets_name=None,
                          datasets_root_path=DEFAULT_DATASETS_ROOT):
    """
    Gets the root path of the dataset. If the dataset is not found, it downloads and extracts it.

    Args:
        dataset_root_path (str, optional): The root path of the dataset. Defaults to None.
        dataset_name (str, optional): Name of the dataset. Defaults to None.
        datasets_name (str, optional): Name of the datasets (either 'UCR' or 'UEA'). Defaults to None.
        datasets_root_path (str, optional): The root path of the datasets. Defaults to DEFAULT_DATASETS_ROOT.

    Returns:
        str: The root path of the dataset.
    """
    if dataset_name is None:
        print("Please provide a valid dataset name.")
        return None
    
    if dataset_root_path is None:
        dataset_root_path = os.path.join(datasets_root_path, dataset_name)
    dataset_root_path = os.path.abspath(dataset_root_path)
    
    ucr_url = 'https://www.timeseriesclassification.com/ClassificationDownloads/Archives/Univariate2018_ts.zip'
    ucr_password = "someone"

    uea_url = "https://www.timeseriesclassification.com/ClassificationDownloads/Archives/Multivariate2018_arff.zip"

    if not os.path.isdir(dataset_root_path):
        os.makedirs(dataset_root_path, exist_ok=True)
        if datasets_name.lower() == "ucr":
            download_file(ucr_url, "UCR", datasets_root_path)

            with zipfile.ZipFile(datasets_root_path + "UCR_TS_Archive_2018.zip", 'r') as zf:
                if zf.setpassword(bytes(ucr_password, 'utf-8')):
                    zf.extractall(dataset_root_path)
                    print("The ZIP file has been successfully extracted. UCR 2018 zip is successfully unzip.")

        elif datasets_name.lower() == "uea":
            download_file(uea_url, "UEA", datasets_root_path)

            with zipfile.ZipFile(datasets_root_path + "Multivariate2018_ts.zip", 'r') as zf:
                zf.extractall(dataset_root_path)
                print("The ZIP file has been successfully extracted. UEA 2018 zip is successfully unzip.")

        else:
            print("the dataset is not contain in our contain, please use the Dataset command")
    else:
        print(dataset_name + " is available")
    return dataset_root_path

def data_file_type(dataset_name,dataset_root_path=None, datasets_root_name=None, all_file_path=None):
    """
    Determines the data file type based on the dataset name, and reads the data from each file in the dataset.

    Args:
        dataset_name (str): Name of the dataset.
        dataset_root_path (str, optional): The root path of the dataset. Defaults to None.
        datasets_root_name (str, optional): Name of the datasets (either 'UCR' or 'UEA'). Defaults to None.
        all_file_path (str, optional): All file path. Defaults to None.

    Returns:
        pandas.DataFrame, pandas.DataFrame: DataFrames containing the training and testing data.
    """
    if datasets_root_name.lower() == "ucr":
        # Define the paths for the training and testing sets.
        train_file_path = os.path.join(dataset_root_path, dataset_name, dataset_name + "_TRAIN.tsv")
        test_file_path = os.path.join(dataset_root_path, dataset_name, dataset_name + "_TEST.tsv")
        # Read the data from each file in the training set and testing set using the pandas library, which is a library for data processing.
        train_file_df = pd.read_csv(train_file_path, sep='\t', header=None)
        test_file_df = pd.read_csv(test_file_path, sep='\t', header=None)

    elif datasets_root_name.lower() == "uea":
        # Define the paths for the training and testing sets.
        train_file_path = os.path.join(dataset_root_path, dataset_name, dataset_name + "_TRAIN.arff")
        test_file_path = os.path.join(dataset_root_path, dataset_name, dataset_name + "_TEST.arff")
        train_data_arff = arff.loadarff(train_file_path)
        train_file_df = pd.DataFrame(train_data_arff[0])
        test_data_arff = arff.loadarff(test_file_path)
        test_file_df = pd.DataFrame(test_data_arff[0])
    return train_file_df, test_file_df



class RandomSplitOpenDataset:
    """
    A class used to represent a dataset that is randomly split into training and testing sets.

    Attributes:
        x_train (numpy.ndarray): The input features for the training set.
        y_train (numpy.ndarray): The labels for the training set.
        x_test (numpy.ndarray): The input features for the testing set.
        y_test (numpy.ndarray): The labels for the testing set.
        train_size_rate (float): The percentage of samples to include in the training set.
        open_label_rate (float): The percentage of unique labels to select for the open set.
        train_random_state (int): Random seed for splitting the data into training and testing sets.
        open_random_state (int): Random seed for selecting open set labels.
    """
    def __init__(
            self,
            x_all,
            y_all,
            train_size_rate=0.3,
            test_size_rate=0.3,
            open_label_rate=0.5,
            train_random_state=42,
            test_random_state=42,
            open_random_state=42
            ):
        
        self.x_all = x_all
        self.y_all = y_all
        self.train_size_rate = train_size_rate
        self.test_size_rate = test_size_rate
        self.open_label_rate = open_label_rate
        self.train_random_state = train_random_state
        self.test_random_state = test_random_state
        self.open_random_state = open_random_state
        
        self.all_features, self.all_labels= x_all.numpy(), y_all.numpy()

    def load(self):
        """
        Loads the dataset and splits it into training and testing sets.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: Tensors containing the input features and labels for the training and testing sets.
        """

        # Get unique labels and calculate the number of labels to select for the test set
        unique_labels = np.unique(self.all_labels)
        y_open_nums = int(math.ceil(len(unique_labels) * self.open_label_rate))

        # # if the number of y_open is 0, it indicates that the dataset has not open data.
        # if y_open_nums == 0:
        #     raise ValueError("The number of open label data is 0, please change the percentage.")

        random.seed(self.train_random_state)
        # choose the y_open and unselected_labels in y_all. the labels in the y_open and unselected_labels are unique. 
        self.selected_open_labels = random.sample(list(unique_labels), y_open_nums)
        unselected_labels = [_ for _ in unique_labels if _ not in self.selected_open_labels]

        self.x_train_val, x_test, self.y_train_val, y_test = train_test_split(self.all_features, self.all_labels, train_size= 1 - self.test_size_rate, random_state=self.train_random_state, stratify=self.all_labels)
        
        x_train, x_val, y_train, y_val = train_test_split(self.x_train_val, self.y_train_val, train_size=self.train_size_rate / (1 - self.test_size_rate), random_state=self.test_random_state, stratify=self.y_train_val)

        train_mask = np.isin(y_train, self.selected_open_labels, invert=True)
        val_mask = np.isin(y_val, self.selected_open_labels, invert=True)
        x_train, x_val = x_train[train_mask], x_val[val_mask]
        y_train, y_val = y_train[train_mask], y_val[val_mask]
        
        x_train, y_train, x_val, y_val, x_test, y_test = torch.tensor(x_train), torch.tensor(y_train).long(), torch.tensor(x_val), torch.tensor(y_val).long(), torch.tensor(x_test), torch.tensor(y_test).long()

        return x_train, y_train, x_val, y_val, x_test, y_test
        
class RandomSplitOpenAllDataset(RandomSplitOpenDataset):
    """
    A dataset class that extends the functionality of RandomSplitOpenDataset by introducing a new partition 
    of the test dataset. This partition specifically contains samples with labels that are part of the "open" set.

    Attributes:
        - All attributes inherited from RandomSplitOpenDataset.

    Methods:
        - __init__: Constructor method.
        - load: Returns the training and test datasets, with an additional partition 
                of the test dataset containing only the "open" labeled data points.

    Usage:
    ```python
    dataset = RandomSplitOpenAllDataset(x_train, y_train, x_test, y_test, train_size_rate, open_label_rate)
    x_train, y_train, x_test, y_test, last_x_test, last_y_test = dataset.load()
    ```

    Note:
        Ensure RandomSplitOpenDataset is defined in the current environment and its `load` method is functional.
    """
    def __init__(self, x_train, y_train, x_test, y_test, train_size_rate, open_label_rate, train_random_state=42, open_random_state=42):
        super().__init__(x_train, y_train, x_test, y_test, train_size_rate, open_label_rate, train_random_state, open_random_state)
    
    def load(self):
        x_train, y_train, x_test, y_test = super().load()
        x_train, y_train, x_test, y_test = x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy() 

        mask = np.isin(y_test, self.selected_open_labels, invert=True)
        unmask = ~mask
        x_test, last_x_test = x_test[mask], x_test[unmask]
        y_test, last_y_test = y_test[mask], y_test[unmask]


        x_train, y_train, x_test, y_test, last_x_test, last_y_test = torch.tensor(x_train), torch.tensor(y_train).long(), torch.tensor(x_test), torch.tensor(y_test).long(), torch.tensor(last_x_test), torch.tensor(last_y_test).long()
        
        return x_train, y_train, x_test, y_test, last_x_test, last_y_test


def relabel_from_zero(*args):
    """
    Relabels input labels starting from zero. 

    This can be useful in scenarios where labels are not consecutive or start from a non-zero 
    number, but a consecutive zero-based labeling is desired.

    Args:
    - *args (tuple): A variable-length argument tuple. Can be one or two torch.Tensors 
      representing the labels to be relabeled.

    Returns:
    - torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Relabeled labels. If one tensor is 
      passed, one is returned. If two tensors are passed, a tuple of two tensors is returned.
    """ 
    if len(args) > 2:
        raise ValueError("Too many arguments. This function expects 1 or 2 arguments.") 
    encoder = LabelEncoder()

    if len(args) == 1:
        labels = args[0].numpy()
        labels = encoder.fit_transform(labels)
        labels = torch.tensor(labels).long()
        return labels
    if len(args) == 2:
        y_train, y_test = args[0].numpy(), args[1].numpy()
        encoder.fit(np.concatenate([y_train, y_test]))
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)
        y_train, y_test = torch.tensor(y_train).long(),torch.tensor(y_test).long()
        return y_train, y_test

def preprocess_test_labels(y_test, y_train, real_label):
    """
    Function to preprocess test labels: 
    replace labels in `real_label` with the corresponding labels in y_train, 
    and assign new labels to the rest

    Args:
    y_test (torch.Tensor): Test labels to be preprocessed
    y_train (torch.Tensor): Training labels used for mapping
    real_label (torch.Tensor): Real labels in training set

    Returns:
    torch.Tensor: Preprocessed test labels
    """

    # Convert to list for easy manipulation
    real_label_list = real_label.tolist()
    y_train_unique = torch.unique(y_train).tolist()

    # Create a mapping from real_label to the corresponding y_train labels
    label_mapping = {real_label_list[i]: y_train_unique[i] for i in range(len(real_label_list))}

    # Maximum label in y_train
    max_train_label = max(y_train_unique)

    def relabel_test(y):
        if y.item() in real_label_list:
            return label_mapping[y.item()]
        else:
            return max_train_label + 1

    y_test_preprocessed = torch.tensor([relabel_test(y) for y in y_test])

    return y_test_preprocessed

def delete_label_and_corresponding_data(x_train, y_train, labels):
    """
    Remove specific data points from the dataset based on their labels.

    Args:
    x_train (torch.Tensor): Input features from the training dataset.
    y_train (torch.Tensor): Labels from the training dataset.
    labels (int or list): Label or list of labels to be removed from the dataset.

    Returns:
    torch.Tensor, torch.Tensor: Filtered input features and labels without the specified label(s).
    """
    all_labels = torch.unique(y_train)
    new_x_train, new_y_train = [], []
    for i, j in zip(y_train, x_train):
        if i != labels:
            new_x_train.append(j)
            new_y_train.append(i)
    new_x_train, new_y_train = torch.stack(new_x_train), torch.stack(new_y_train)
    return new_x_train, new_y_train


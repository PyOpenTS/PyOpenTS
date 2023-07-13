import os
import requests
import zipfile 
import pandas as pd
from scipy.io import arff
import numpy as np
import random
from sklearn.model_selection import train_test_split

DEFAULT_DATASETS_ROOT = "data"

def download_file(url, datasets_name, datasets_root_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(datasets_root_path, 'wb') as file:
            file.write(response.content)
        print(datasets_name + "is successfully downloaded")
    else:
        print(datasets_name + "is not successfully downloaded")


def get_dataset_root_path(dataset_root_path=None, dataset_name=None, datasets_name=None,
                          datasets_root_path=DEFAULT_DATASETS_ROOT):

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
        print(dataset_name + "is available")
    return dataset_root_path

def data_file_type(dataset_name,dataset_root_path=None, datasets_root_name=None, all_file_path=None):

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


def split_train_test(x_all, y_all, train_percentage=0.5, open_percentage=0.5, open_random=42, train_random_state=42):

    # Get unique labels and calculate the number of labels to select for the test set
    y_unique = np.unique(y_all)
    y_open_nums = int(len(y_unique) * open_percentage)

    # if the number of y_open is 0, it indicates that the dataset has not open data.
    if y_open_nums == 0:
        raise ValueError("The number of open label data is 0, please change the percentage.")

    # fix the seed of random
    random.seed(open_random)

    # choose the y_open and y_not_open in y_all. the labels in the y_open and y_not_open are unique. 
    y_open_select = random.sample(list(y_unique), y_open_nums)
    y_not_open = [_ for _ in y_unique if _ not in y_open_select]

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size=train_percentage,random_state=train_random_state, stratify=y_all)

    mask = np.isin(y_train, y_open_select, invert=True)
    x_train = x_train[mask]
    y_train = y_train[mask]

    return x_train, y_train, x_test, y_test
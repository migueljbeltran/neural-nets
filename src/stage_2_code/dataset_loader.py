'''
Concrete IO class for stage 2 MNIST dataset (CSV format)
label, pixel1, pixel2, ..., pixel784
'''

import numpy as np

from src.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_description = dDescription

    def load(self):
        print('loading training data...')
        train_data = np.loadtxt(
            self.dataset_source_folder_path + self.dataset_train_file_name,
            delimiter=','
        )
        print('loading testing data...')
        test_data = np.loadtxt(
            self.dataset_source_folder_path + self.dataset_test_file_name,
            delimiter=','
        )

        # First column is label, rest 784 are features
        X_train = train_data[:, 1:].astype(np.float32) / 255.0  # normalize to [0,1]
        y_train = train_data[:, 0].astype(np.int64)

        X_test = test_data[:, 1:].astype(np.float32) / 255.0
        y_test = test_data[:, 0].astype(np.int64)

        self.data = {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test}
        }
        print(f'Train size: {X_train.shape}, Test size: {X_test.shape}')
        return self.data
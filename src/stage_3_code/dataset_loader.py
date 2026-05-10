'''
Concrete dataset loader for stage 3 image datasets.
'''

import pickle
import numpy as np

from src.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None, *,
                 use_single_channel=False,
                 normalize=False):
        super().__init__(dName, dDescription)
        self.use_single_channel = use_single_channel
        self.normalize = normalize

    def _convert_images(self, instances):
        X, y = [], []
        for item in instances:
            image = np.asarray(item['image'], dtype=np.float32)

            if image.ndim == 2:
                image = image[np.newaxis, :, :]
            elif self.use_single_channel:
                image = image[:, :, 0][np.newaxis, :, :]
            else:
                image = np.transpose(image, (2, 0, 1))

            X.append(image / 255.0)
            y.append(int(item['label']))

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

    def load(self):
        print(f'loading data from {self.dataset_source_file_name}...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            loaded = pickle.load(f)

        X_train, y_train = self._convert_images(loaded['train'])
        X_test, y_test = self._convert_images(loaded['test'])

        # ORL labels are 1..40; shift to 0..39 for CrossEntropyLoss.
        min_label = int(min(y_train.min(), y_test.min()))
        if min_label != 0:
            y_train -= min_label
            y_test -= min_label

        if self.normalize:
            mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
            std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        self.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
        }
        print(f'train shape: {X_train.shape}, test shape: {X_test.shape}')
        return self.data

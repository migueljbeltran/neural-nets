'''
Concrete dataset loader for stage 3 image datasets.
'''

import pickle
import numpy as np

from src.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    use_single_channel = False

    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_description = dDescription

    def _convert_images(self, instances):
        X = []
        y = []
        for item in instances:
            image = np.array(item['image'], dtype=np.float32)
            label = int(item['label'])

            if image.ndim == 2:
                image = image[np.newaxis, :, :]
            elif image.ndim == 3:
                if self.use_single_channel:
                    image = image[:, :, 0]
                    image = image[np.newaxis, :, :]
                else:
                    image = np.transpose(image, (2, 0, 1))

            image = image / 255.0
            X.append(image)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def load(self):
        print(f'loading data from {self.dataset_source_file_name}...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            loaded = pickle.load(f)

        X_train, y_train = self._convert_images(loaded['train'])
        X_test, y_test = self._convert_images(loaded['test'])

        # ORL labels are 1..40; convert to 0..39 for CrossEntropyLoss.
        min_label = int(min(y_train.min(), y_test.min()))
        if min_label != 0:
            y_train = y_train - min_label
            y_test = y_test - min_label

        self.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
        }
        print(f'train shape: {X_train.shape}, test shape: {X_test.shape}')
        return self.data

'''
Result saver for Stage 4 experiments.
'''

import os
import pickle

from src.base_class.result import result


class Result_Saver(result):
    def save(self):
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name,
        )
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name,
        )
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        return self.data

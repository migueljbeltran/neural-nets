'''
Result saver for stage 3 experiments.
'''

import pickle
import os

from src.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        with open(self.result_destination_folder_path + self.result_destination_file_name, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        return

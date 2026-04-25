import pickle
import os

from src.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def __init__(self, rName=None, rDescription=None):
        self.result_name = rName
        self.result_description = rDescription

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        path = self.result_destination_folder_path + self.result_destination_file_name
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        pass
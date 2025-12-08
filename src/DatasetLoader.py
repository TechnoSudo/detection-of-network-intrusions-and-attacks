import pandas as pd


class DatasetLoader:

    def __init__(self, dataset_1_path, dataset_2_path):
        self.dataset_1_path = dataset_1_path
        self.dataset_2_path = dataset_2_path

        self.df1 = pd.read_csv(self.dataset_1_path)
        self.df2 = pd.read_csv(self.dataset_2_path, header=None)

    def load_dataset_1(self, path=None):
        file_path = path or self.dataset_1_path
        return pd.read_csv(file_path)

    def load_dataset_2(self, path=None):
        file_path = path or self.dataset_2_path
        return pd.read_csv(file_path, header=None)

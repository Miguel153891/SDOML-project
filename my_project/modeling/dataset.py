import pandas as pd
import torch
from torch.utils.data import Dataset

class AutomobileDataset(Dataset):
    def __init__(self, path):
        df = pd.read_parquet(path)

        self.X = torch.tensor(
            df.drop(columns=["Selling_Price"]).values,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            df["Selling_Price"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
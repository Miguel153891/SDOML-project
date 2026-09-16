"""Dataset utilities for the automobile price prediction model."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class AutomobileDataset(Dataset):
    """PyTorch dataset for automobile price prediction.

    Loads a processed Parquet dataset, separates the input features from
    the target variable, and converts both to PyTorch tensors.

    Parameters
    ----------
    path : str
        Path to the Parquet file containing the processed automobile data.

    Attributes
    ----------
    X : torch.Tensor
        Input features as a two-dimensional float32 tensor.
    y : torch.Tensor
        Target selling prices as a one-dimensional float32 tensor.

    Examples
    --------
    >>> dataset = AutomobileDataset("data/processed/automobile_dataset.parquet")
    >>> len(dataset)
    # Number of samples in the dataset
    """

    def __init__(self, path: str):
        """Initialize the automobile dataset.

        Parameters
        ----------
        path : str
            Path to the Parquet dataset.
        """
        df = pd.read_parquet(path)

        self.X = torch.tensor(
            df.drop(columns=["Selling_Price"]).values,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            df["Selling_Price"].values,
            dtype=torch.float32
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples available in the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the input features and target selling price.
        """
        return self.X[index], self.y[index]
"""Training utilities for the automobile price prediction model."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_project.dataset import AutomobileDataset


def create_model(input_size: int) -> nn.Module:

    """Create the neural network used for price prediction.

    Parameters
    ----------
    input_size : int
        Number of input features.

    Returns
    -------
    torch.nn.Module
        Neural network for predicting automobile selling prices.

    """

    return nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001
) -> nn.Module:

    """Train the automobile price prediction model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network to train.

    train_loader : torch.utils.data.DataLoader
        DataLoader containing the training data.

    epochs : int, default=100
        Number of training epochs.

    learning_rate : float, default=0.001
        Learning rate used by the Adam optimizer.

    Returns
    -------
    torch.nn.Module
        The trained model.

    """

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):

        for X, y in train_loader:
            prediction = model(X).squeeze()
            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model


def main():
    """Train the automobile price prediction model and save it to disk."""

    PROCESSED_TRAIN_FILE = "data/processed/automobile_dataset"
    PROCESSED_TEST_FILE = "data/processed/automobile_test"
    MODEL_PATH = Path("models/automobile_model.pth")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = AutomobileDataset(
        f"{PROCESSED_TRAIN_FILE}.parquet"
    )

    test_dataset = AutomobileDataset(
        f"{PROCESSED_TEST_FILE}.parquet"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    # The test loader is prepared here for later evaluation.
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    model = create_model(train_dataset.X.shape[1])

    model = train_model(
        model,
        train_loader
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
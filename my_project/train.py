"""Training utilities for the automobile price prediction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_project.dataset import AutomobileDataset


def create_model(input_size):
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


def train_model(model, train_loader, epochs=100, learning_rate=0.001):
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
    train_dataset = AutomobileDataset(
        "data/processed/automobile_dataset.parquet"
    )

    test_dataset = AutomobileDataset(
        "data/processed/automobile_test.parquet"
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
        "models/automobile_model.pth"
    )

    print("Model saved to models/automobile_model.pth")


if __name__ == "__main__":
    main()
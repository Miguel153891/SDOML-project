from pathlib import Path

import torch
from torch.utils.data import DataLoader

from my_project.dataset import AutomobileDataset
from my_project.train import (
    create_model,
    train_model_with_history
)
from my_project.evaluation import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "automobile_dataset.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "automobile_model.pth"
)


def train_from_interface(
    epochs=50,
    learning_rate=0.001,
    batch_size=32
):
    """
    Train a model using hyperparameters provided
    by the Gradio interface.
    """

    epochs = int(epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)

    # Load dataset
    dataset = AutomobileDataset(
        str(TRAIN_FILE)
    )

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create model
    model = create_model(
        dataset.X.shape[1]
    )

    # Train
    model, history = train_model_with_history(
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )

    # Save checkpoint
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    # Final training loss
    final_loss = history[-1]

    return (
        history,
        final_loss,
        str(MODEL_PATH)
    )
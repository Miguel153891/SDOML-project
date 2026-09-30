import numpy as np
import torch

from my_project.dataset import AutomobileDataset
from my_project.train import create_model
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def evaluate_model():

    dataset = AutomobileDataset(
        PROJECT_ROOT / "data" / "processed" / "automobile_test.parquet"
    )
    model = create_model(
        dataset.X.shape[1]
    )

    #model.load_state_dict(
    #    torch.load(
    #        "../models/automobile_model.pth",
    #        map_location="cpu"
    #    )
    #)

    model.load_state_dict(
        torch.load(
            PROJECT_ROOT / "models" / "automobile_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X, y in dataset:

            prediction = model(
                X.unsqueeze(0)
            ).item()

            predictions.append(prediction)
            targets.append(y.item())

    predictions = np.array(predictions)
    targets = np.array(targets)

    errors = predictions - targets

    metrics = {
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "r2": 1 - (
            np.sum(errors ** 2)
            / np.sum((targets - targets.mean()) ** 2)
        )
    }

    return targets, predictions, metrics
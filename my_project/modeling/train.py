import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_project.modeling.dataset import AutomobileDataset

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

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Model
input_size = train_dataset.X.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for X, y in train_loader:
        prediction = model(X).squeeze()
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


torch.save(
    model.state_dict(),
    "models/automobile_model.pth"
)

print("Model saved to models/automobile_model.pth")
from pathlib import Path

import pandas as pd
import torch

from my_project.train import create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "automobile_model.pth"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "automobile_dataset.parquet"


def predict_price(year: int, engine_size: float, mileage: int, horsepower: float,
    torque: float, owners: int, accident_history: float, service_history: str,
    fuel_efficiency: float, make: str, model_name: str, fuel_type: str,
    transmission: str, color: str, body_type: str, drivetrain: str,
    location: str) -> float:
    """
    Train model and return loss history.
        
    Parameters
    ----------
    year: int
        Data necessary for the prediction, representing one feature with its same name.
    engine_size: float
        Data necessary for the prediction, representing one feature with its same name.
    mileage: int
        Data necessary for the prediction, representing one feature with its same name.
    horsepower: float
        Data necessary for the prediction, representing one feature with its same name.
    torque: float
        Data necessary for the prediction, representing one feature with its same name.
    owners: int
        Data necessary for the prediction, representing one feature with its same name.
    accident_history: float
        Data necessary for the prediction, representing one feature with its same name.
    service_history: str
        Data necessary for the prediction, representing one feature with its same name.
    fuel_efficiency: float
        Data necessary for the prediction, representing one feature with its same name.
    make: str
        Data necessary for the prediction, representing one feature with its same name.
    model_name: str
        Data necessary for the prediction, representing one feature with its same name.
    fuel_type: str
        Data necessary for the prediction, representing one feature with its same name.
    transmission: str
        Data necessary for the prediction, representing one feature with its same name.
    color: str
        Data necessary for the prediction, representing one feature with its same name.
    body_type: str
        Data necessary for the prediction, representing one feature with its same name.
    drivetrain: str
        Data necessary for the prediction, representing one feature with its same name.
    location: str
        Data necessary for the prediction, representing one feature with its same name.
        
    Returns
    -------
    prediction : float
        the selling price predicted
    """
    
    # Load processed training data to obtain
    # exactly the same feature columns.
    df = pd.read_parquet(DATA_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column != "Selling_Price"
    ]

    # Create an empty row with all required features.
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical features
    input_data["Year"] = year
    input_data["Engine_Size"] = engine_size
    input_data["Mileage"] = mileage
    input_data["Horsepower"] = horsepower
    input_data["Torque"] = torque
    input_data["Owners"] = owners
    input_data["Accident_History"] = accident_history
    input_data["Service_History"] = service_history
    input_data["Fuel_Efficiency"] = fuel_efficiency

    # One-hot categorical features
    categorical_values = {
        "Make": make,
        "Model": model_name,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Color": color,
        "Body_Type": body_type,
        "Drivetrain": drivetrain,
        "Location": location,
    }

    for prefix, value in categorical_values.items():

        column_name = f"{prefix}_{value}"

        if column_name in input_data.columns:
            input_data[column_name] = 1

    # Convert to tensor
    X = torch.tensor(
        input_data.values,
        dtype=torch.float32
    )

    # Create model
    model = create_model(
        input_data.shape[1]
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    # Prediction
    with torch.no_grad():
        prediction = model(X).item()

    return prediction
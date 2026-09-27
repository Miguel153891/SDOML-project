import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def car_model_count(df):
    car_counts = df["Make"].value_counts().reset_index()
    car_counts.columns = ["Make", "Count"]

    fig = px.bar(
        car_counts,
        x="Make",
        y="Count",
        title="Automobile Dataset: Car Model Count",
        labels={
            "Make": "Car Model",
            "Count": "Count"
        }
    )

    fig.update_layout(
        xaxis_tickangle=-90
    )

    return fig


def correlation_plot(df):
    corr = df.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    return fig


def scatterplots(df):
    fig = px.scatter(
        df,
        x="Selling_Price",
        y="Mileage",
        color="Make",
        title="Selling Price vs Mileage",
        labels={
            "Selling_Price": "Selling Price",
            "Mileage": "Mileage",
            "Make": "Make"
        },
        hover_data=df.columns
    )

    return fig


def selling_vs_car(df):
    fig = px.box(
        df,
        x="Make",
        y="Selling_Price",
        title="Selling Price vs Car",
        labels={
            "Make": "Car Model",
            "Selling_Price": "Selling Price"
        }
    )

    fig.update_layout(
        xaxis_tickangle=-90
    )

    return fig


def pairplots(df):
    numeric_columns = df.select_dtypes(include=np.number).columns

    fig = px.scatter_matrix(
        df,
        dimensions=numeric_columns,
        title="Automobile Dataset: Pairplot"
    )

    return fig


def actual_vs_predicted(targets, predictions):
    """Plot actual prices against predicted prices."""

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        targets,
        predictions,
        alpha=0.6
    )

    min_value = min(
        targets.min(),
        predictions.min()
    )

    max_value = max(
        targets.max(),
        predictions.max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted")

    return fig


def residual_distribution(targets, predictions):
    """Plot the distribution of prediction errors."""

    residuals = predictions - targets

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        residuals,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")

    return fig


def residuals_vs_predictions(targets, predictions):
    """Plot residuals against predicted prices."""

    residuals = predictions - targets

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        predictions,
        residuals,
        alpha=0.6
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predictions")

    return fig

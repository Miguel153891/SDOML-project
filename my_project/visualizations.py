import numpy as np
import plotly
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def car_model_count(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """
    Makes a plot showing how many cars each car model has.
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the data
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Bar plot showing the car amounts per car model
    """
    
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


def correlation_plot(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """
    Creates a plot heatmap showing the correlation of features.
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the data
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot showing a correlation heatmap
    """
    
    corr = df.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    return fig


def scatterplots(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """
    Creates a scatterplot showing selling price and Mileage against each other.
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the data
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Scatterplot between selling price and Mileage
    """
    
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


def selling_vs_car(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """
    Creates a boxplot of the car model and the selling price.
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the data
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Boxplot of the car model on selling price
    """
    
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


def pairplots(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """
    Creates all pairplots of the features.
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the data
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Pairplots of the features
    """
    
    numeric_columns = df.select_dtypes(include=np.number).columns

    fig = px.scatter_matrix(
        df,
        dimensions=numeric_columns,
        title="Automobile Dataset: Pairplot"
    )

    return fig


def actual_vs_predicted(targets: np.ndarray, predictions: np.ndarray) -> plotly.graph_objects.Figure:
    """
    Plot actual prices against predicted prices.
        
    Parameters
    ----------
    targets : np.ndarray
        True targets from the dataset
    predictions : np.ndarray
        Predicted targets on the dataset
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot showing the regression and the scattered data
    """

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


def residual_distribution(targets: np.ndarray, predictions: np.ndarray) -> plotly.graph_objects.Figure:
    """
    Plot the distribution of prediction errors.
        
    Parameters
    ----------
    targets : np.ndarray
        True targets from the dataset
    predictions : np.ndarray
        Predicted targets on the dataset
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot showing the prediction errors
    """

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


def residuals_vs_predictions(targets: np.ndarray, predictions: np.ndarray) -> plotly.graph_objects.Figure:
    """
    Plot residuals against predicted prices.
        
    Parameters
    ----------
    targets : np.ndarray
        True targets from the dataset
    predictions : np.ndarray
        Predicted targets on the dataset
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plot showing the residuals and the scattered data
    """

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

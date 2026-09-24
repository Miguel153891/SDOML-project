import numpy as np
import plotly.express as px


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
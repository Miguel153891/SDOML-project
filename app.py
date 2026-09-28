import gradio as gr
import pandas as pd
import sys
from pathlib import Path


# 1. PROJECT PATH

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# 2. IMPORT PROJECT FUNCTIONS

from my_project.preprocessing import prepare_dataset

from my_project.fileManager import FileManager

from my_project.visualizations import (
    car_model_count,
    correlation_plot,
    scatterplots,
    selling_vs_car,
    pairplots,
    actual_vs_predicted,
    residual_distribution,
    residuals_vs_predictions
)

from my_project.evaluation import evaluate_model
from my_project.prediction import predict_price
from my_project.training_interface import train_from_interface


# 3. DATASET

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "automobile_dataset"

fM = FileManager()
print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_FILE:", RAW_FILE)
print("CSV PATH:", Path(str(RAW_FILE) + ".csv"))
print("CSV EXISTS:", Path(str(RAW_FILE) + ".csv").exists())
# Prepare dataset automatically if it does not exist
if not Path(str(RAW_FILE) + ".csv").exists():
    prepare_dataset()

fM.set_format("csv")
df = fM.read(str(RAW_FILE))


# 4. MODEL EVALUATION

targets = None
predictions = None

metrics = {
    "mae": 0,
    "rmse": 0,
    "r2": 0
}

MODEL_PATH = PROJECT_ROOT / "models" / "automobile_model.pth"

if MODEL_PATH.exists():
    targets, predictions, metrics = evaluate_model()


# 5. GRADIO THEME

custom_theme = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[
        "Inter",
        "ui-sans-serif",
        "system-ui",
        "sans-serif"
    ],
    font_mono=[
        "ui-monospace",
        "SFMono-Regular",
        "Menlo",
        "monospace"
    ]
).set(
    body_background_fill="#f4f5f7",
    body_background_fill_dark="#111315",

    block_background_fill="#ffffff",
    block_background_fill_dark="#191c1f",

    block_border_width="1px",
    block_border_color="#e1e4e8",
    block_border_color_dark="#30343a",

    block_radius="10px",

    input_background_fill="#ffffff",
    input_background_fill_dark="#191c1f",

    button_primary_background_fill="#263238",
    button_primary_background_fill_hover="#37474f",
    button_primary_text_color="#ffffff",

    body_text_color="#202428",
    body_text_color_dark="#f1f3f5",

    block_label_text_color="#5f6368",
    block_label_text_color_dark="#aeb4bb"
)


# ============================================================
# 6. CUSTOM CSS
# ============================================================

custom_css = """

/* ----------------------------------------------------------
   GENERAL
---------------------------------------------------------- */

body {
    background: #f4f5f7;
}

.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 0 35px 40px 35px !important;
}


/* ----------------------------------------------------------
   MAIN HEADER
---------------------------------------------------------- */
.main-header {
    padding: 55px 10px 42px 10px;
    border-bottom: 1px solid #dfe3e8;
    margin-bottom: 30px;
    text-align: center;
}

.main-header .eyebrow {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #687078;
    margin-bottom: 12px;
}

.main-header h1 {
    font-size: 42px;
    line-height: 1.1;
    font-weight: 650;
    letter-spacing: -1.5px;
    color: #17191c;
    margin: 0 0 14px 0;
}

.main-header .subtitle {
    max-width: 720px;
    margin: 0 auto;
    font-size: 17px;
    line-height: 1.6;
    color: #626970;
}

.main-header .authors {
    margin-top: 32px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.2px;
    color: #737b82;
}

.main-header .authors span {
    margin: 0 10px;
    color: #b0b5ba;
}

/* ----------------------------------------------------------
   SECTION HEADERS
---------------------------------------------------------- */

.section-header {
    margin: 10px 0 22px 0;
}

.section-header .section-number {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #7a828a;
    margin-bottom: 7px;
}

.section-header h2 {
    font-size: 27px;
    font-weight: 650;
    letter-spacing: -0.5px;
    color: #1c2024;
    margin: 0 0 7px 0;
}

.section-header p {
    font-size: 14px;
    color: #70777e;
    margin: 0;
}


/* ----------------------------------------------------------
   TABS
---------------------------------------------------------- */

button[role="tab"] {
    font-size: 14px !important;
    font-weight: 600 !important;
}

button[role="tab"][aria-selected="true"] {
    color: #263238 !important;
}


/* ----------------------------------------------------------
   METRIC BOXES
---------------------------------------------------------- */

.metric-box {
    background: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 10px;
    padding: 4px;
}

.metric-box input {
    font-size: 24px !important;
    font-weight: 650 !important;
}


/* ----------------------------------------------------------
   PREDICTION FORM
---------------------------------------------------------- */

.prediction-panel {
    background: #ffffff;
    border: 1px solid #dfe3e8;
    border-radius: 12px;
    padding: 28px !important;
}


/* ----------------------------------------------------------
   FORM SECTION TITLES
---------------------------------------------------------- */

.form-section {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #596168;
    border-bottom: 1px solid #e5e7ea;
    padding-bottom: 10px;
    margin: 18px 0 18px 0;
}


/* ----------------------------------------------------------
   ESTIMATED PRICE
---------------------------------------------------------- */

.result-panel {
    background: #ffffff !important;
    border: 1px solid #dfe3e8 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-top: 24px;
}


/* Label */

.result-panel label {
    color: #202428 !important;
    font-weight: 600 !important;
}


/* Input / displayed value */

.result-panel input {
    background: #ffffff !important;
    border: none !important;
    color: #202428 !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}


/* ----------------------------------------------------------
   PREDICT BUTTON
---------------------------------------------------------- */

#predict-button {
    height: 52px;
    border-radius: 7px !important;
    font-size: 15px !important;
    font-weight: 650 !important;
    margin-top: 15px;
}


/* ----------------------------------------------------------
   PLOTS
---------------------------------------------------------- */

.plot-container {
    border-radius: 10px !important;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer {
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #dfe3e8;
    color: #858c92;
    font-size: 12px;
}

"""


# 7. GRADIO APP

with gr.Blocks() as app:

    # HEADER
    gr.HTML(
        """
        <div class="main-header">
    
            <div class="eyebrow">
                SOFTWARE DEVELOPMENT ORIENTED TO MACHINE LEARNING
            </div>
    
            <h1>
                Automobile Market Analytics
            </h1>
    
            <div class="subtitle">
                Automobile price prediction using a model
                trained on vehicle specifications, technical
                characteristics and market information.
            </div>
    
            <div class="authors">
                Miguel Zubitur
                <span>·</span>
                Magdalena Hristova
                <span>·</span>
                Diego Suarez
            </div>
    
        </div>
        """
    )


    # ========================================================
    # TAB 1 — DATA EXPLORATION
    # ========================================================

    with gr.Tab("Data Exploration"):

        gr.HTML(
            """
            <div class="section-header">

                <div class="section-number">
                    01 — DATA
                </div>

                <h2>
                    Dataset exploration
                </h2>

                <p>
                    Visual analysis of the automobile dataset and
                    relationships between vehicle characteristics
                    and selling price.
                </p>

            </div>
            """
        )


        # ----------------------------------------------------
        # Dataset download and preprocessing
        # ----------------------------------------------------

        with gr.Row():

            download_button = gr.Button(
                "Download & Prepare Dataset",
                variant="primary"
            )

            dataset_status = gr.Textbox(
                label="Dataset Status",
                interactive=False
            )

        def download_and_prepare():

            train_df, test_df = prepare_dataset()

            return (
                "Dataset downloaded and preprocessed successfully."
            )


        download_button.click(
            fn=download_and_prepare,
            inputs=[],
            outputs=dataset_status
        )
            
        # ----------------------------------------------------
        # Dataset sample and basic statistics
        # ----------------------------------------------------

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("### Dataset Sample")

                gr.Dataframe(
                    value=df.head(10),
                    interactive=False
                )

            with gr.Column(scale=1):

                gr.Markdown("### Basic Statistics")

                basic_statistics = pd.DataFrame({
                    "Variable": df.columns,
                    "Type": [
                        str(df[column].dtype)
                        for column in df.columns
                    ],
                    "Non-null": [
                        df[column].notna().sum()
                        for column in df.columns
                    ],
                    "Unique": [
                        df[column].nunique()
                        for column in df.columns
                    ],
                    "Mean": [
                        round(df[column].mean(), 2)
                        if pd.api.types.is_numeric_dtype(df[column])
                        else None
                        for column in df.columns
                    ],
                    "Min": [
                        round(df[column].min(), 2)
                        if pd.api.types.is_numeric_dtype(df[column])
                        else None
                        for column in df.columns
                    ],
                    "Max": [
                        round(df[column].max(), 2)
                        if pd.api.types.is_numeric_dtype(df[column])
                        else None
                        for column in df.columns
                    ]
                })

                gr.Dataframe(
                    value=basic_statistics,
                    interactive=False,
                    wrap=True
                )

        # ----------------------------------------------------
        # Dataset overview
        # ----------------------------------------------------

        gr.Markdown("### Dataset Overview")

        with gr.Row():

            with gr.Column(elem_classes="metric-box"):

                gr.Number(
                    value=len(df),
                    label="Total Observations",
                    interactive=False
                )

            with gr.Column(elem_classes="metric-box"):

                gr.Number(
                    value=len(df.columns),
                    label="Total Features",
                    interactive=False
                )

            with gr.Column(elem_classes="metric-box"):

                gr.Number(
                    value=df.isna().sum().sum(),
                    label="Missing Values",
                    interactive=False
                )

            with gr.Column(elem_classes="metric-box"):

                gr.Number(
                    value=df["Selling_Price"].mean(),
                    label="Average Selling Price",
                    precision=2,
                    interactive=False
                )
                
        # Row 1
        gr.Markdown("### Visualizations")

        with gr.Row():

            with gr.Column(scale=1):

                gr.Plot(
                    value=car_model_count(df),
                    label="Car Model Count"
                )

            with gr.Column(scale=1):

                gr.Plot(
                    value=correlation_plot(df),
                    label="Correlation Heatmap"
                )


        # Row 2
        with gr.Row():

            with gr.Column(scale=1):

                gr.Plot(
                    value=scatterplots(df),
                    label="Scatterplots"
                )

            with gr.Column(scale=1):

                gr.Plot(
                    value=selling_vs_car(df),
                    label="Selling Price vs Car"
                )


        # Pairplot
        gr.Plot(
            value=pairplots(df),
            label="Pairplot"
        )

    # ========================================================
    # TAB 2 — TRAINING
    # ========================================================
    
    with gr.Tab("Training"):
    
        gr.HTML(
            """
            <div class="section-header">
    
                <div class="section-number">
                    02 — TRAINING
                </div>
    
                <h2>
                    Model training
                </h2>
    
                <p>
                    Configure the main training hyperparameters
                    and train the model.
                </p>
    
            </div>
            """
        )
    
    
        # ----------------------------------------------------
        # Training configuration
        # ----------------------------------------------------
    
        with gr.Column(
            elem_classes="prediction-panel"
        ):
    
            gr.HTML(
                '<div class="form-section">Training Configuration</div>'
            )
    
    
            with gr.Row():
    
                epochs_input = gr.Number(
                    label="Epochs",
                    value=50,
                    minimum=1,
                    maximum=500,
                    step=1
                )
    
                learning_rate_input = gr.Number(
                    label="Learning Rate",
                    value=0.001,
                    minimum=0.00001,
                    maximum=0.1,
                    step=0.0001
                )
    
                batch_size_input = gr.Number(
                    label="Batch Size",
                    value=32,
                    minimum=1,
                    maximum=256,
                    step=1
                )


        gr.Markdown(
            """
            Adjust the hyperparameters above and start a new
            training run using the processed training dataset.
            """
        )


        train_button = gr.Button(
            "Train Model",
            variant="primary",
            elem_id="predict-button"
        )

    
        # ----------------------------------------------------
        # Training results
        # ----------------------------------------------------
    
        gr.HTML(
            '<div class="form-section">Training Results</div>'
        )
    
    
        with gr.Row():

            with gr.Column(
                elem_classes="metric-box"
            ):
        
                final_loss_output = gr.Number(
                    label="Final Training Loss",
                    interactive=False
                )
        
            with gr.Column(
                elem_classes="metric-box"
            ):
        
                checkpoint_output = gr.Textbox(
                    label="Saved Checkpoint",
                    interactive=False
                )
            
    
        # ----------------------------------------------------
        # Training loss
        # ----------------------------------------------------
    
        gr.Markdown(
            "### Training Loss"
        )
    
    
        training_plot = gr.Plot(
            label="Loss per Epoch"
        )
    
    
        # ----------------------------------------------------
        # Training button
        # ----------------------------------------------------
    
        def run_training(
            epochs,
            learning_rate,
            batch_size
        ):
        
            history, final_loss, checkpoint = train_from_interface(
                epochs=int(epochs),
                learning_rate=float(learning_rate),
                batch_size=int(batch_size)
            )
        
            import matplotlib.pyplot as plt
        
            fig, ax = plt.subplots(
                figsize=(8, 4)
            )
        
            ax.plot(
                range(1, len(history) + 1),
                history
            )
        
            ax.set_xlabel(
                "Epoch"
            )
        
            ax.set_ylabel(
                "Training Loss"
            )
        
            ax.set_title(
                "Training Loss per Epoch"
            )
        
            ax.grid(
                alpha=0.2
            )
        
            plt.tight_layout()
        
            return (
                fig,
                round(final_loss, 4),
                checkpoint
            )
    
    
        train_button.click(
    
            fn=run_training,
    
            inputs=[
                epochs_input,
                learning_rate_input,
                batch_size_input
            ],
    
            outputs=[
                training_plot,
                final_loss_output,
                checkpoint_output
            ]
        )

   # ========================================================
    # TAB 3 — MODEL EVALUATION
    # ========================================================
    
    with gr.Tab("Model Evaluation"):
    
        gr.HTML(
            """
            <div class="section-header">
    
                <div class="section-number">
                    03 — MODEL
                </div>
    
                <h2>
                    Model performance
                </h2>
    
                <p>
                    Evaluation of the trained model on the
                    automobile test dataset.
                </p>
    
            </div>
            """
        )
    
        evaluate_button = gr.Button(
            "Evaluate Trained Model",
            variant="primary"
        )
    
        evaluation_status = gr.Textbox(
            label="Evaluation Status",
            interactive=False
        )
    
        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
    
        with gr.Row():
    
            with gr.Column(elem_classes="metric-box"):
    
                mae_output = gr.Number(
                    label="Mean Absolute Error (MAE)",
                    interactive=False
                )
    
            with gr.Column(elem_classes="metric-box"):
    
                rmse_output = gr.Number(
                    label="Root Mean Squared Error (RMSE)",
                    interactive=False
                )
    
            with gr.Column(elem_classes="metric-box"):
    
                r2_output = gr.Number(
                    label="Coefficient of Determination (R²)",
                    interactive=False
                )
    
        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------
    
        gr.Markdown(
            "### Prediction diagnostics"
        )
    
        with gr.Row():
    
            with gr.Column():
    
                actual_predicted_plot = gr.Plot(
                    label="Actual vs Predicted"
                )
    
            with gr.Column():
    
                residual_distribution_plot = gr.Plot(
                    label="Residual Distribution"
                )
    
        with gr.Row():
    
            with gr.Column():
    
                residual_predictions_plot = gr.Plot(
                    label="Residuals vs Predictions"
                )
    
    
        # ----------------------------------------------------
        # Evaluation function
        # ----------------------------------------------------
    
        def run_evaluation():
    
            model_path = PROJECT_ROOT / "models" / "automobile_model.pth"
    
            if not model_path.exists():
    
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "No trained model available. Train the model first."
                )
    
            targets, predictions, metrics = evaluate_model()
    
            return (
                round(metrics["mae"], 2),
                round(metrics["rmse"], 2),
                round(metrics["r2"], 3),
                actual_vs_predicted(
                    targets,
                    predictions
                ),
                residual_distribution(
                    targets,
                    predictions
                ),
                residuals_vs_predictions(
                    targets,
                    predictions
                ),
                "Model evaluated successfully."
            )
    
    
        evaluate_button.click(
    
            fn=run_evaluation,
    
            inputs=[],
    
            outputs=[
                mae_output,
                rmse_output,
                r2_output,
                actual_predicted_plot,
                residual_distribution_plot,
                residual_predictions_plot,
                evaluation_status
            ]
        )


    # ========================================================
    # TAB 4 — PRICE PREDICTION
    # ========================================================

    with gr.Tab("Price Prediction"):

        gr.HTML(
            """
            <div class="section-header">

                <div class="section-number">
                    04 — PREDICTION
                </div>

                <h2>
                    Estimate vehicle price
                </h2>

                <p>
                    Enter the characteristics of a vehicle to
                    obtain an estimated selling price from the
                    trained model.
                </p>

            </div>
            """
        )


        # ----------------------------------------------------
        # Prediction form
        # ----------------------------------------------------

        with gr.Column(
            elem_classes="prediction-panel"
        ):


            # =================================================
            # VEHICLE
            # =================================================

            gr.HTML(
                '<div class="form-section">Vehicle</div>'
            )


            with gr.Row():

                make = gr.Dropdown(
                    choices=[
                        "Audi",
                        "BMW",
                        "Chevrolet",
                        "Ford",
                        "Honda",
                        "Hyundai",
                        "Mercedes-Benz",
                        "Nissan",
                        "Toyota",
                        "Volkswagen"
                    ],
                    label="Make",
                    value="Toyota"
                )


                model_name = gr.Dropdown(
                    choices=[
                        "3 Series",
                        "5 Series",
                        "A4",
                        "A6",
                        "Accord",
                        "Altima",
                        "Atlas",
                        "C-Class",
                        "CR-V",
                        "Camry",
                        "Civic",
                        "Corolla",
                        "E-Class",
                        "Elantra",
                        "Equinox",
                        "Escape",
                        "Explorer",
                        "F-150",
                        "GLC",
                        "GLE",
                        "Golf",
                        "Highlander",
                        "Malibu",
                        "Mustang",
                        "Passat",
                        "Pathfinder",
                        "Pilot",
                        "Q5",
                        "Q7",
                        "RAV4",
                        "Rogue",
                        "Santa Fe",
                        "Sentra",
                        "Silverado",
                        "Sonata",
                        "Tahoe",
                        "Tiguan",
                        "Tucson",
                        "X3",
                        "X5"
                    ],
                    label="Model",
                    value="Corolla"
                )


            with gr.Row():

                year = gr.Number(
                    label="Year",
                    value=2020
                )

                mileage = gr.Number(
                    label="Mileage",
                    value=50000
                )

                owners = gr.Number(
                    label="Number of Owners",
                    value=1
                )


            # =================================================
            # PERFORMANCE
            # =================================================

            gr.HTML(
                '<div class="form-section">Performance</div>'
            )


            with gr.Row():

                engine_size = gr.Number(
                    label="Engine Size",
                    value=2.0
                )

                horsepower = gr.Number(
                    label="Horsepower",
                    value=150
                )

                torque = gr.Number(
                    label="Torque",
                    value=150
                )

                fuel_efficiency = gr.Number(
                    label="Fuel Efficiency",
                    value=25
                )


            # =================================================
            # VEHICLE CONDITION
            # =================================================

            gr.HTML(
                '<div class="form-section">Vehicle Condition</div>'
            )


            with gr.Row():

                accident_history = gr.Number(
                    label="Accident History",
                    value=0
                )

                service_history = gr.Number(
                    label="Service History",
                    value=1
                )


            # =================================================
            # CONFIGURATION
            # =================================================

            gr.HTML(
                '<div class="form-section">Configuration</div>'
            )


            with gr.Row():

                fuel_type = gr.Dropdown(
                    choices=[
                        "Diesel",
                        "Electric",
                        "Hybrid",
                        "Petrol"
                    ],
                    label="Fuel Type",
                    value="Petrol"
                )


                transmission = gr.Dropdown(
                    choices=[
                        "Automatic",
                        "Manual"
                    ],
                    label="Transmission",
                    value="Automatic"
                )


                color = gr.Dropdown(
                    choices=[
                        "Black",
                        "Blue",
                        "Brown",
                        "Gray",
                        "Green",
                        "Red",
                        "Silver",
                        "White"
                    ],
                    label="Color",
                    value="Black"
                )


            with gr.Row():

                body_type = gr.Dropdown(
                    choices=[
                        "Coupe",
                        "Hatchback",
                        "SUV",
                        "Sedan",
                        "Truck"
                    ],
                    label="Body Type",
                    value="Sedan"
                )


                drivetrain = gr.Dropdown(
                    choices=[
                        "4WD",
                        "AWD",
                        "FWD",
                        "RWD"
                    ],
                    label="Drivetrain",
                    value="FWD"
                )


                location = gr.Dropdown(
                    choices=[
                        "CA",
                        "FL",
                        "GA",
                        "IL",
                        "MI",
                        "NC",
                        "NY",
                        "OH",
                        "PA",
                        "TX"
                    ],
                    label="Location",
                    value="CA"
                )


            # =================================================
            # RESULT
            # =================================================

            gr.HTML(
                '<div class="form-section">Estimated Market Value</div>'
            )


            predict_button = gr.Button(
                "Estimate price",
                variant="primary",
                elem_id="predict-button"
            )


            prediction_output = gr.Number(
                label="Estimated Selling Price",
                interactive=False,
                elem_classes="result-panel"
            )


            # =================================================
            # BUTTON ACTION
            # =================================================

            predict_button.click(

                fn=predict_price,

                inputs=[
                    year,
                    engine_size,
                    mileage,
                    horsepower,
                    torque,
                    owners,
                    accident_history,
                    service_history,
                    fuel_efficiency,
                    make,
                    model_name,
                    fuel_type,
                    transmission,
                    color,
                    body_type,
                    drivetrain,
                    location
                ],

                outputs=prediction_output
            )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

            Automobile Market Analytics
            &nbsp;·&nbsp;
            Price Prediction
            &nbsp;·&nbsp;
            SDOML Project

        </div>
        """
    )


# ============================================================
# 8. LAUNCH
# ============================================================

#app.launch(
#    css=custom_css,
#    share=True
#)

app.launch(
    css=custom_css
)
<h1 align="center">SDOML Project</h1>

<p align="center">
  <strong>Automobile Market Analytics</strong>
</p>

<h2 align="center">Team #5</h2>

<details>
<summary><strong>Team members</strong></summary>

* Miguel Zubitur
* Magdalena Hristova
* Diego Suarez

</details>

---

## Project Information

<details>
<summary><strong>What does this project do?</strong></summary>

This project was developed for the **Software Development Oriented to Machine Learning (SDOML)** course.

The project analyzes the factors that influence the price of used cars, including vehicle characteristics such as age, mileage, condition, and technical features.

A machine learning model based on a neural network is also developed to predict the selling price of an automobile from its characteristics.

The project includes:

* Data downloading
* Data preprocessing
* Exploratory data analysis
* Machine learning model training
* Model performance analysis

</details>

<details>
<summary><strong>Project documentation</strong></summary>

The project documentation is published using **GitHub Pages**:

**[Project Documentation](https://miguel153891.github.io/SDOML-project/)**

The documentation is generated using `pdoc` and is published from the `docs/` directory.

</details>

---

## Dataset

<details>
<summary><strong>Data source</strong></summary>

The dataset used in this project was obtained from Kaggle:

[Automobile Market Analytics Dataset](https://www.kaggle.com/datasets/deeplumiere/automobile-market-analytics-dataset)

The dataset is downloaded automatically using `kagglehub`.

The dataset is **not stored in the repository**. It can be downloaded again by following the workflow described below.

</details>

---

## Installation

<details>
<summary><strong>Requirements</strong></summary>

The project requires:

* Python 3.10 or newer
* [uv](https://docs.astral.sh/uv/)
* Jupyter Notebook

The Python dependencies are managed through `pyproject.toml` and `uv.lock`.

</details>

<details>
<summary><strong>Install the project</strong></summary>

Clone the repository:

```bash
git clone https://github.com/Miguel153891/SDOML-project.git
cd SDOML-project
```

Install the project dependencies:

```bash
uv sync
```

This creates the required environment and installs the dependencies specified by the project.

</details>

---

## How to Run the Project

The complete workflow should be executed in the following order.

<details>
<summary><strong>1. Download the dataset</strong></summary>

Run:

```bash
uv run python -m my_project.download_data
```

This downloads the dataset from Kaggle and creates:

```text
data/
└── raw/
    └── automobile_dataset.csv
```

The `data/raw/` directory is generated automatically if it does not exist.

</details>

<details>
<summary><strong>2. Process and explore the data</strong></summary>

Open:

```text
notebooks/data_exploration.ipynb
```

and execute all cells.

The notebook performs the data preprocessing and exploratory data analysis.

The processed datasets are generated in:

```text
data/
└── processed/
    ├── automobile_dataset.parquet
    └── automobile_test.parquet
```

The exploratory analysis also generates figures in:

```text
reports/
└── figures/
    └── exploration/
```

</details>

<details>
<summary><strong>3. Train the model</strong></summary>

After the processed datasets have been generated, run:

```bash
uv run python -m my_project.train
```

The training script creates and trains the neural network.

The trained model is saved to:

```text
models/
└── automobile_model.pth
```

The `models/` directory is created automatically if it does not exist.

</details>

<details>
<summary><strong>4. Analyze model performance</strong></summary>

Open:

```text
notebooks/performance_analysis.ipynb
```

and execute all cells.

The notebook loads the trained model and analyzes its predictions and performance.

</details>

---

## Complete Workflow

For a clean setup, execute:

```bash
uv sync
```

Then download the dataset:

```bash
uv run python -m my_project.download_data
```

Then execute all cells in:

```text
notebooks/data_exploration.ipynb
```

Then train the model:

```bash
uv run python -m my_project.train
```

Finally, execute:

```text
notebooks/performance_analysis.ipynb
```

The complete workflow is:

```text
Kaggle
   │
   ▼
download_data.py
   │
   ▼
data/raw/
   │
   ▼
data_exploration.ipynb
   │
   ├──► reports/figures/exploration/
   │
   ▼
data/processed/
   │
   ▼
train.py
   │
   ▼
models/automobile_model.pth
   │
   ▼
performance_analysis.ipynb
```

---

## Project Structure

<details>
<summary><strong>Repository structure</strong></summary>

```text
SDOML-project/
│
├── my_project/
│   ├── __init__.py
│   ├── dataset.py
│   ├── download_data.py
│   ├── fileManager.py
│   └── train.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── performance_analysis.ipynb
│
├── reports/
│   ├── export_notebooks/
│   └── figures/
│
├── docs/
│   └── Generated project documentation
│
├── LICENSE
├── README.md
├── Requirements.md
├── pyproject.toml
├── uv.lock
├── practice1.pdf
├── practice2.pdf
└── practice3.pdf
```

The following directories are generated locally and are not committed to Git:

```text
data/
├── raw/
└── processed/

models/
```

</details>

---

## Main Components

<details>
<summary><strong>Python modules</strong></summary>

### `my_project/download_data.py`

Downloads the automobile dataset from Kaggle using `kagglehub`.

### `my_project/dataset.py`

Defines the PyTorch `AutomobileDataset`, which loads the processed Parquet data and converts it into PyTorch tensors.

### `my_project/train.py`

Contains the neural network and training functionality:

* `create_model()` — creates the neural network.
* `train_model()` — trains the model.
* `main()` — loads the datasets, trains the model, and saves the trained model.

</details>

<details>
<summary><strong>Jupyter notebooks</strong></summary>

### `data_exploration.ipynb`

Performs:

* Data loading
* Data preprocessing
* Exploratory data analysis
* Dataset preparation
* Generation of exploratory figures

### `performance_analysis.ipynb`

Loads the trained model and evaluates its predictions and performance.

</details>

---

## Documentation Generation

<details>
<summary><strong>Generate documentation locally</strong></summary>

The API documentation is generated using [pdoc](https://pdoc.dev/).

Run the command
```bash
uv run pdoc -o docs my_project --docformat numpy
```

The generated documentation is stored in `docs/` and published automatically through GitHub Pages.

</details>

---

## Contributing

<details>
<summary><strong>Contribution guidelines</strong></summary>

Contributions should follow the project's Git workflow.

1. Create a feature branch from `main`:

```bash
git checkout -b feature/my-change
```

2. Make the required changes.
3. Test the changes locally.
4. Use descriptive commit messages:

```bash
git add .
git commit -m "Add descriptive change"
```

5. Push the feature branch:

```bash
git push origin feature/my-change
```

6. Open a pull request to `main`.

Changes should be focused and documented when appropriate.

Do not commit:

* Downloaded datasets
* Processed datasets
* Trained model files
* Python cache files
* Other generated files excluded by `.gitignore`

</details>

---

## License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for the complete license text.

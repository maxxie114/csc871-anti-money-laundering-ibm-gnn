# Anti-Money Laundering Detection

This project, developed for **CSC871 - Deep Learning**, explores various machine learning approaches to detect money laundering activities in financial transactions using the IBM Anti-Money Laundering (AML) dataset. It includes multiple experiments ranging from Graph Neural Networks (GNNs) to traditional tabular models like MLP and XGBoost.

## Authors
- Andy Byeon
- Chris Randall
- Max Xie

## Table of Contents
- [Project Structure](#project-structure)
- [Models and Experiments](#models-and-experiments)
  - [GNN: TransformerConv (Main)](#gnn-transformerconv-main)
  - [GNN: GINEConv](#gnn-gineconv)
  - [Tabular Baseline: MLP](#tabular-baseline-mlp)
  - [Tabular Baseline: XGBoost](#tabular-baseline-xgboost)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Utilities](#utilities)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
.
├───.gitattributes
├───.gitignore
├───.python-version
├───anti_money_laundering_ibm.ipynb
├───cuda_diagnose.py
├───gine-model.ipynb
├───gine-with-batches-model.ipynb
├───LICENSE
├───pyproject.toml
├───README.md
├───requirements.txt
├───tabular_mlp.ipynb
├───tabular_xgboost.ipynb
├───uv.lock
├───dataset/
│   ├───HI-Small_accounts.csv
│   └───HI-Small_Trans.csv
└───plots/
    ├───best_f1_mlp.png
    ├───roc_pr_mlp.png
    └───target_fpr_mlp.png
```

## Models and Experiments

This repository contains several Jupyter notebooks, each representing a different modeling approach to the AML problem.

### GNN: TransformerConv (Main)
- **File**: `anti_money_laundering_ibm.ipynb`
- **Approach**: This is the primary and most developed experiment. It models the transaction data as a graph and uses a Graph Neural Network with `TransformerConv` layers from PyTorch Geometric.
- **Features**:
    - **Data Splitting**: Stratified 60/20/20 split to maintain the class imbalance ratio across train, validation, and test sets.
    - **Training**: A sophisticated training loop that supports batched training on edge chunks. It includes a fallback mechanism for environments without `pyg-lib` or `torch-sparse`.
    - **Evaluation**: Implements dynamic threshold calibration based on a target False Positive Rate (FPR) on the validation set, which is crucial for imbalanced datasets.

### GNN: GINEConv
- **Files**: `gine-model.ipynb`, `gine-with-batches-model.ipynb`
- **Approach**: These notebooks explore an alternative GNN architecture using Graph Isomorphism Network with Edge features (`GINEConv`).
- **Features**:
    - **Feature Engineering**: Uses one-hot encoding for categorical node and edge features.
    - **Data Splitting**: Employs a chronological 60/20/20 split, simulating a more realistic time-based evaluation.
    - **Training**:
        - `gine-model.ipynb`: Trains on the full graph at once.
        - `gine-with-batches-model.ipynb`: An enhanced version that uses `LinkNeighborLoader` for memory-efficient batched training.

### Tabular Baseline: MLP
- **File**: `tabular_mlp.ipynb`
- **Approach**: This notebook serves as a strong non-GNN baseline. It ignores the graph structure and treats the problem as a traditional tabular classification task.
- **Features**:
    - **Feature Engineering**: Creates a rich set of features, including time-based features (hour, day of week), transaction-specific features (e.g., `same_bank`, `amount_ratio`), and more.
    - **Model**: A Multi-Layer Perceptron (MLP) built with PyTorch, using embedding layers to handle categorical features.
    - **Evaluation**: Provides a comprehensive evaluation framework, analyzing model performance at multiple decision thresholds (default 0.5, best F1-score, and a target FPR).

### Tabular Baseline: XGBoost
- **File**: `tabular_xgboost.ipynb`
- **Approach**: A second tabular baseline that uses the powerful XGBoost library, a state-of-the-art gradient boosting model.
- **Features**:
    - **Feature Engineering**: Uses the exact same feature set as the MLP notebook for a fair comparison.
    - **Model**: An `XGBClassifier` tuned for the imbalanced dataset, using `scale_pos_weight` to handle the class distribution.
    - **Preprocessing**: Uses integer encoding for categorical features, which is efficient for tree-based models.

## Dataset

The project uses the **IBM HI-Small** dataset, which consists of two main files:

- **`dataset/HI-Small_accounts.csv`**: Contains information about bank accounts.
- **`dataset/HI-Small_Trans.csv`**: Contains transactional data between accounts, with a label indicating whether a transaction is part of a money laundering scheme.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/csc865-anti-money-laundering-ibm.git
    cd csc865-anti-money-laundering-ibm
    ```

2.  **Set up the project environment**:
    This project uses `uv` and `pyproject.toml` for dependency management. If you don't have `uv`, install it first:
    ```bash
    pip install uv
    ```
    Then, create a virtual environment and install all dependencies:
    ```bash
    uv sync
    ```
    To activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```
    Note: The dependencies in `pyproject.toml` are configured for a CUDA 12.8 environment. If you have a different CUDA version or are using a CPU, `uv` will attempt to find compatible packages. If you encounter issues, you might need to manually adjust the PyTorch installation.

## Usage

All experiments are contained within Jupyter Notebooks (`.ipynb` files). To run an experiment:

1.  **Start a Jupyter server**:
    ```bash
    jupyter notebook
    ```

2.  **Open a notebook**:
    Navigate to and open one of the main notebooks, such as `anti_money_laundering_ibm.ipynb` or `tabular_mlp.ipynb`.

3.  **Run the cells**:
    Execute the cells in the notebook sequentially to load the data, preprocess it, train the model, and evaluate the results.

## Utilities

- **`cuda_diagnose.py`**: A simple script to help diagnose your PyTorch and CUDA installation. Run it to ensure your environment is set up correctly for GPU training:
  ```bash
  python cuda_diagnose.py
  ```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
# csc865-anti-money-laundering-ibm
A GNN project on anti-money laundering using IBM's transaction dataset

# Anti-Money Laundering with Graph Neural Networks

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to build a Graph Neural Network (GNN) for analyzing the IBM money laundering dataset. The goal is to detect patterns and anomalies in financial transactions that may indicate money laundering activities. The project is being developed entirely in Jupyter Notebook files, leveraging PyTorch and TorchMetrics for model implementation.

## Dataset
We are currently using the HI-Small dataset, which includes:
- **Nodes**: Representing account numbers.
- **Edges**: Representing transactions between accounts.
- **Node Labels**: None.
- **Edge Labels**: A tensor with two labels, one for sent transactions and one for received transactions.
- **Encoding**: One-hot encoding for labels.

Future iterations may involve larger datasets for more comprehensive analysis.

## Model Details
We are starting with a basic Graph Convolutional Network (GCN) with the following characteristics:
- **Nodes**: Represent account numbers.
- **Edges**: Represent transactions.
- **Node Labels**: None.
- **Edge Labels**: A tensor with two labels, one for sent transactions and one for received transactions.
- **Encoding**: One-hot encoding for labels.

Future enhancements may include exploring Hypergraph Convolution (HypergraphConv) for advanced modeling.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: PyTorch, TorchMetrics, Pandas, Matplotlib
- **Tools**: Jupyter Notebook

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/maxxie114/csc865-anti-money-laundering-ibm.git
   ```

2. Navigate to the project directory:
   ```bash
   cd csc865-anti-money-laundering-ibm
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure that you have a CUDA-compatible GPU and the necessary drivers installed for PyTorch.

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `anti_money_laundering_ibm.ipynb` file and run the cells sequentially.

3. The notebook includes data loading, preprocessing, and model training steps.

## Future Plans
- Explore larger datasets for more comprehensive analysis.
- Investigate the use of Hypergraph Convolution (HypergraphConv) for advanced modeling.
- Optimize the model for better performance and scalability.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

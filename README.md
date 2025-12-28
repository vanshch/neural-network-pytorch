# Iris Classification Neural Network

A PyTorch-based neural network implementation for classifying iris flowers using the classic Iris dataset.

## Overview

This project implements a feedforward neural network to classify iris flowers into three species (Setosa, Versicolor, and Virginica) based on four features: sepal length, sepal width, petal length, and petal width.

## Features

- **ETL Pipeline**: Custom Extract-Transform-Load (ETL) class for data preprocessing
- **Neural Network**: Three-layer feedforward network with ReLU activation
- **Training Loop**: 100 epochs with Adam optimizer and CrossEntropy loss
- **Reproducibility**: Fixed random seed for consistent results

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
torch
```

Install dependencies using:
```bash
pip install numpy pandas matplotlib scikit-learn torch
```

## Dataset

The program automatically fetches the Iris dataset from:
```
https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/iris.csv
```

**Dataset Features:**
- 150 samples
- 4 input features (sepal length, sepal width, petal length, petal width)
- 3 output classes (species: setosa, versicolor, virginica)

## Model Architecture

```
Input Layer:    4 neurons  (features)
Hidden Layer 1: 8 neurons  (ReLU activation)
Hidden Layer 2: 9 neurons  (ReLU activation)
Output Layer:   3 neurons  (class probabilities)
```

## Usage

Run the program:
```bash
python iris_classifier.py
```

The program will:
1. Download and load the Iris dataset
2. Preprocess the data (encode species labels, split train/test)
3. Train the neural network for 100 epochs
4. Print loss every 10 epochs

**Expected Output:**
```
epoch 0 and loss 1.234...
epoch 10 and loss 0.987...
epoch 20 and loss 0.765...
...
```

## Code Structure

### Classes

**`Model(nn.Module)`**
- PyTorch neural network with configurable layers
- Default: 4 input → 8 hidden → 9 hidden → 3 output
- Uses ReLU activation functions

**`ELT`**
- `extract()`: Downloads dataset from URL
- `transform()`: Encodes categorical labels and separates features/targets
- `load()`: Splits data into train/test sets and converts to PyTorch tensors

### Hyperparameters

- **Epochs**: 100
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Train/Test Split**: 80/20
- **Random Seed**: 41

## Customization

You can modify the model architecture by changing the initialization parameters:

```python
model = Model(
    inp=4,           # input features
    hid_layer1=16,   # first hidden layer size
    hid_layer2=16,   # second hidden layer size
    output=3,        # output classes
    random_seed=41   # reproducibility seed
)
```

## Future Improvements

- Add model evaluation on test set
- Implement accuracy metrics
- Save/load trained model
- Visualize loss curves
- Add confusion matrix
- Implement early stopping
- Add model validation during training

## License

This project uses the public Iris dataset, which is freely available for educational and research purposes.

## Author

Created as an educational example of PyTorch neural network implementation for multi-class classification.
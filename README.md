# Neural Network From Scratch (NumPy Only)

A fully connected neural network implemented entirely from scratch using **NumPy only**.

This project rebuilds the core mechanics of deep learning without TensorFlow, PyTorch, or any ML frameworks. Every step — forward propagation, backpropagation, and gradient descent — is written manually to help you understand how neural networks work under the hood.

---

## Overview

The model trains on an MNIST-style handwritten digit dataset where:

- Each image is 28 × 28 pixels
- Flattened into 784 input features
- Classified into digits 0–9

All gradients, weight updates, and activations are computed manually using NumPy.

This project is ideal for learning the mathematical foundations of neural networks.

---

## Model Architecture

```
Input (784)
   ↓
Dense (10) + ReLU
   ↓
Dense (10) + Softmax
   ↓
Output (10 classes)
```

| Layer  | Units | Activation |
|--------|----------|-------------|
| Input  | 784      | —           |
| Hidden | 10       | ReLU        |
| Output | 10       | Softmax     |

---

## Features

- Pure NumPy implementation
- No deep learning frameworks
- Manual forward propagation
- Manual backpropagation
- Gradient descent optimization
- Vectorized operations
- Train/validation split
- Pixel normalization
- Accuracy evaluation
- Digit visualization using matplotlib
- Educational and beginner-friendly code

---

## Project Structure

```
.
├── Neural_Network_From_Scratch.ipynb   # main implementation
├── train.csv                          # dataset
└── README.md
```

---

## Dataset Format

The dataset is stored in `train.csv`.

Each row:

```
label, pixel1, pixel2, ..., pixel784
```

- label → digit class (0–9)
- pixels → grayscale values (0–255)

---

## Installation

Install dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

---

## Usage

Launch Jupyter Notebook:

```bash
jupyter notebook Neural_Network_From_Scratch.ipynb
```

Run all cells.

The notebook will automatically:

1. Load dataset
2. Shuffle data
3. Split into training and validation sets
4. Normalize inputs
5. Train the neural network
6. Evaluate accuracy
7. Display predictions

---

## Core Functions

### Parameter Initialization
```python
initialize_parameters()
```

### Forward Propagation
```python
forward_propagation()
```

### Activation Functions
```python
ReLU()
softmax()
```

### Training (Gradient Descent)
```python
gradient_descent()
```

### Evaluation
```python
get_predictions()
get_accuracy()
```

---

## Example Output

```
Validation Accuracy: 91%
Predicted: 7
Actual: 7
```

The notebook also visualizes the corresponding handwritten digit.

---

## Concepts Covered

This project reinforces understanding of:

- Linear algebra in neural networks
- Matrix multiplication
- Backpropagation mathematics
- Gradient descent optimization
- Vectorization with NumPy
- Multi-class classification
- Softmax probabilities

---

## Possible Improvements

- Add multiple hidden layers
- Implement mini-batch gradient descent
- Add learning rate scheduling
- Plot training loss curves
- Save and load trained weights
- Add confusion matrix
- Regularization (L2/Dropout)
- Compare with PyTorch or TensorFlow
- GPU acceleration

---

## Learning Goal

Build neural networks from first principles to deeply understand how deep learning works internally.

No black boxes. Just math and NumPy.

---

## License

MIT License

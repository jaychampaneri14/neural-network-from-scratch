# Neural Network From Scratch

A fully functional neural network implemented using only NumPy — no deep learning frameworks.

## Implemented Features
- He and Xavier weight initialization
- Activations: ReLU, Sigmoid, Tanh, Linear, Softmax
- Losses: Cross-entropy (classification), MSE (regression)
- Adam optimizer with bias correction
- Dropout regularization
- L2 weight regularization
- Mini-batch gradient descent
- Forward and backward propagation

## Demo Tasks
1. **Moons dataset** — binary classification with decision boundary
2. **Iris dataset** — 3-class classification
3. **Regression** — fitting a damped sine function

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `moons_boundary.png` — decision boundary + loss curve
- `iris_training.png` — accuracy over epochs
- `regression_fit.png` — function approximation

## Architecture Example
```python
nn = NeuralNetwork([784, 128, 64, 10], ['relu', 'relu', 'softmax'])
nn.fit(X_train, Y_train, epochs=100, lr=0.001)
```

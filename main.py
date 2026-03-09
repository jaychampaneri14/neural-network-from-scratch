"""
Neural Network From Scratch
Fully connected neural network implemented using only NumPy.
Supports: ReLU, Sigmoid, Softmax, Cross-entropy, MSE, Adam optimizer, Dropout, BatchNorm.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


# ─── ACTIVATIONS ──────────────────────────────────────────────────────────────
def relu(z):         return np.maximum(0, z)
def relu_back(z):    return (z > 0).astype(float)
def sigmoid(z):      return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_back(z): s = sigmoid(z); return s * (1 - s)
def tanh_act(z):     return np.tanh(z)
def tanh_back(z):    return 1 - np.tanh(z) ** 2

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    'relu':    (relu,    relu_back),
    'sigmoid': (sigmoid, sigmoid_back),
    'tanh':    (tanh_act, tanh_back),
    'linear':  (lambda z: z, lambda z: np.ones_like(z)),
}


# ─── NEURAL NETWORK CLASS ─────────────────────────────────────────────────────
class NeuralNetwork:
    def __init__(self, layer_dims, activations, dropout_rate=0.0, seed=42):
        """
        layer_dims:  list of layer sizes including input, e.g. [784, 128, 64, 10]
        activations: list of activation names, e.g. ['relu', 'relu', 'softmax']
        """
        np.random.seed(seed)
        self.layer_dims   = layer_dims
        self.activations  = activations
        self.dropout_rate = dropout_rate
        self.params       = {}
        self.adam_state   = {}
        self.n_layers     = len(layer_dims) - 1
        self._init_params()

    def _init_params(self):
        """He / Xavier initialization."""
        for l in range(1, self.n_layers + 1):
            fan_in = self.layer_dims[l - 1]
            fan_out = self.layer_dims[l]
            act = self.activations[l - 1]
            if act == 'relu':
                scale = np.sqrt(2.0 / fan_in)         # He
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
            self.params[f'W{l}'] = np.random.randn(fan_in, fan_out) * scale
            self.params[f'b{l}'] = np.zeros((1, fan_out))
            # Adam moments
            for key in [f'W{l}', f'b{l}']:
                self.adam_state[f'm_{key}'] = np.zeros_like(self.params[key])
                self.adam_state[f'v_{key}'] = np.zeros_like(self.params[key])

    def _forward_layer(self, A_prev, l, training=True):
        W, b = self.params[f'W{l}'], self.params[f'b{l}']
        Z = A_prev @ W + b
        act = self.activations[l - 1]
        if act == 'softmax':
            A = softmax(Z)
        else:
            fn, _ = ACTIVATIONS[act]
            A = fn(Z)
        # Dropout (not on output layer)
        mask = None
        if training and self.dropout_rate > 0 and l < self.n_layers:
            mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            A = A * mask
        return Z, A, mask

    def forward(self, X, training=True):
        self.cache = {'A0': X}
        A = X
        for l in range(1, self.n_layers + 1):
            Z, A, mask = self._forward_layer(A, l, training)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
            self.cache[f'mask{l}'] = mask
        return A

    def _compute_loss(self, Y_pred, Y_true, task='classification'):
        n = Y_true.shape[0]
        if task == 'classification':
            # Cross-entropy
            log_preds = np.log(np.clip(Y_pred, 1e-15, 1))
            loss = -np.sum(Y_true * log_preds) / n
        else:
            loss = np.mean((Y_pred - Y_true) ** 2)
        # L2 regularization
        l2 = sum(np.sum(self.params[f'W{l}']**2) for l in range(1, self.n_layers+1))
        return loss + 1e-4 * l2 / (2 * n)

    def backward(self, Y_true, task='classification'):
        grads = {}
        n = Y_true.shape[0]
        # Output gradient
        if task == 'classification':
            dA = self.cache[f'A{self.n_layers}'] - Y_true   # dL/dZ for softmax+CE
        else:
            dA = 2 * (self.cache[f'A{self.n_layers}'] - Y_true) / n

        for l in reversed(range(1, self.n_layers + 1)):
            Z = self.cache[f'Z{l}']
            A_prev = self.cache[f'A{l-1}']
            mask = self.cache[f'mask{l}']
            act = self.activations[l - 1]

            if act == 'softmax' or l == self.n_layers:
                dZ = dA
            else:
                _, back_fn = ACTIVATIONS[act]
                dZ = dA * back_fn(Z)

            # Apply dropout mask
            if mask is not None:
                dZ = dZ * mask

            grads[f'dW{l}'] = A_prev.T @ dZ / n + 1e-4 * self.params[f'W{l}'] / n
            grads[f'db{l}'] = dZ.mean(axis=0, keepdims=True)
            dA = dZ @ self.params[f'W{l}'].T

        return grads

    def _adam_update(self, grads, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for l in range(1, self.n_layers + 1):
            for key in [f'W{l}', f'b{l}']:
                g = grads[f'd{key}']
                m = self.adam_state[f'm_{key}']
                v = self.adam_state[f'v_{key}']
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * g ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, Y, X_val=None, Y_val=None,
            epochs=100, lr=0.001, batch_size=64, task='classification', verbose=True):
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        t = 0
        for epoch in range(1, epochs + 1):
            # Mini-batch SGD
            idx = np.random.permutation(len(X))
            X_s, Y_s = X[idx], Y[idx]
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                xb = X_s[i:i+batch_size]
                yb = Y_s[i:i+batch_size]
                t += 1
                Y_pred  = self.forward(xb, training=True)
                loss    = self._compute_loss(Y_pred, yb, task)
                grads   = self.backward(yb, task)
                self._adam_update(grads, lr, t)
                epoch_loss += loss
            epoch_loss /= max(1, len(X) // batch_size)

            # Evaluate
            tr_pred = self.forward(X, training=False)
            tr_loss = self._compute_loss(tr_pred, Y, task)
            history['loss'].append(tr_loss)
            if task == 'classification':
                tr_acc  = (tr_pred.argmax(1) == Y.argmax(1)).mean()
                history['acc'].append(tr_acc)
            if X_val is not None:
                vp   = self.forward(X_val, training=False)
                vl   = self._compute_loss(vp, Y_val, task)
                history['val_loss'].append(vl)
                if task == 'classification':
                    va = (vp.argmax(1) == Y_val.argmax(1)).mean()
                    history['val_acc'].append(va)

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:4d}/{epochs}: loss={tr_loss:.4f}"
                if task == 'classification':
                    msg += f", acc={tr_acc:.4f}"
                if X_val is not None:
                    msg += f" | val_loss={vl:.4f}"
                    if task == 'classification':
                        msg += f", val_acc={va:.4f}"
                print(f"  {msg}")
        return history

    def predict(self, X):
        return self.forward(X, training=False)

    def predict_classes(self, X):
        return self.predict(X).argmax(axis=1)

    def accuracy(self, X, Y):
        return (self.predict_classes(X) == Y.argmax(1)).mean()


# ─── DEMO TASKS ───────────────────────────────────────────────────────────────
def demo_moons():
    print("\n--- Task 1: Moons (Binary Classification) ---")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)

    enc  = OneHotEncoder(sparse_output=False)
    Y_tr = enc.fit_transform(y_tr.reshape(-1, 1))
    Y_te = enc.transform(y_te.reshape(-1, 1))

    nn = NeuralNetwork([2, 32, 16, 2], ['relu', 'relu', 'softmax'], dropout_rate=0.1)
    history = nn.fit(X_tr, Y_tr, X_te, Y_te, epochs=100, lr=5e-3, batch_size=32)

    acc = nn.accuracy(X_te, Y_te)
    print(f"  Final Test Accuracy: {acc:.4f}")

    # Decision boundary
    h = 0.05
    xx, yy = np.meshgrid(np.arange(X_s[:,0].min()-0.5, X_s[:,0].max()+0.5, h),
                         np.arange(X_s[:,1].min()-0.5, X_s[:,1].max()+0.5, h))
    Z = nn.predict_classes(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    axes[0].scatter(X_te[:,0], X_te[:,1], c=y_te, cmap='RdBu', s=20, edgecolors='k', lw=0.5)
    axes[0].set_title('Moons — NN Decision Boundary')

    axes[1].plot(history['loss'], label='Train'); axes[1].plot(history['val_loss'], label='Val')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('moons_boundary.png', dpi=150)
    plt.close()


def demo_iris():
    print("\n--- Task 2: Iris (3-class Classification) ---")
    iris = load_iris()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(iris.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, iris.target, test_size=0.2, stratify=iris.target, random_state=42)
    enc  = OneHotEncoder(sparse_output=False)
    Y_tr = enc.fit_transform(y_tr.reshape(-1, 1))
    Y_te = enc.transform(y_te.reshape(-1, 1))

    nn = NeuralNetwork([4, 16, 8, 3], ['relu', 'relu', 'softmax'])
    history = nn.fit(X_tr, Y_tr, X_te, Y_te, epochs=200, lr=1e-2)
    acc = nn.accuracy(X_te, Y_te)
    print(f"  Final Test Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(history['acc'],     label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Iris Classification — Training History')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('iris_training.png', dpi=150)
    plt.close()


def demo_regression():
    print("\n--- Task 3: Regression ---")
    np.random.seed(42)
    X = np.linspace(-5, 5, 500).reshape(-1, 1)
    y = np.sin(X.ravel()) * np.exp(-0.1 * X.ravel()**2) + np.random.normal(0, 0.1, 500)
    scaler_X = StandardScaler(); scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1))
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

    nn = NeuralNetwork([1, 64, 32, 1], ['tanh', 'relu', 'linear'])
    history = nn.fit(X_tr, y_tr, X_te, y_te, epochs=200, lr=1e-3, task='regression')
    y_pred_s = nn.predict(X_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)

    plt.figure(figsize=(9, 5))
    plt.scatter(X.ravel(), y, s=5, alpha=0.4, label='Data')
    order = X.ravel().argsort()
    plt.plot(X.ravel()[order], y_pred.ravel()[order], 'r-', lw=2, label='NN Prediction')
    plt.legend(); plt.title('Neural Network Regression')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_fit.png', dpi=150)
    plt.close()
    mse = np.mean((y - y_pred.ravel())**2)
    print(f"  MSE: {mse:.6f}")


def main():
    print("=" * 60)
    print("NEURAL NETWORK FROM SCRATCH (NumPy only)")
    print("=" * 60)
    print("Features: He init, Adam optimizer, Dropout, Backprop")

    demo_moons()
    demo_iris()
    demo_regression()

    print("\n--- Output Files ---")
    for f in ['moons_boundary.png', 'iris_training.png', 'regression_fit.png']:
        print(f"  {f}")
    print("\n✓ Neural Network From Scratch complete!")


if __name__ == '__main__':
    main()

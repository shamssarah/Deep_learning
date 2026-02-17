import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

try:
    from hw1_learning_curves import save_learning_curve
except ImportError:
    save_learning_curve = None


def load_fashion_mnist():
    print("Loading Fashion-MNIST...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, cache=True)
    X = X.to_numpy().astype('float32')
    y = y.to_numpy().astype('int32')
    X = X / 255.0  # Normalize to [0,1]
    X = X * 2 - 1   # Scale to [-1,1]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def append_ones(X):
    """Append a column of ones to the right side of matrix X (to incorporate bias into weight matrices)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


class NumpyDenseResMLP:
    """Three-layer DenseNet-style MLP with Layer Normalization.

    Architecture:
        Layer 1: Z1 = AppendOnes(X) @ W1, N1 = LN(Z1), H1 = ReLU(N1)
        Layer 2: C1 = [X || H1], Z2 = AppendOnes(C1) @ W2, N2 = LN(Z2 + H1), H2 = ReLU(N2)
        Layer 3: C2 = [X || H1 || H2], Z3 = AppendOnes(C2) @ W3, N3 = LN(Z3 + H2), H3 = ReLU(N3)
        Output:  Z4 = AppendOnes(H3) @ W4, Y_hat = LogSoftmax(Z4)
        Loss:    NLL = -(1/B) * sum_n Y_hat[n, y_n]
    """

    def __init__(self, input_size, hidden_size, num_classes=10, use_layernorm=True):
        # Your code here - initialize weight matrices and LayerNorm parameters.
        # We suggest He initialization for weights: W ~ N(0, 2/fan_in).
        # Bias is incorporated via AppendOnes, so each W has an extra row.
        # Weight shapes: W1: (d+1, h), W2: (d+h+1, h), W3: (d+2h+1, h), W4: (h+1, C)
        # LayerNorm: gamma = ones(h), beta = zeros(h), one pair per hidden layer.
        raise NotImplementedError

    def layer_norm(self, z, gamma, beta):
        """Forward pass of Layer Normalization (per sample, across hidden dim).

        Returns: (out, cache) where cache = (z_hat, std, gamma) for backward.
        """
        eps = 1e-5
        # Your code here
        raise NotImplementedError

    def layer_norm_backward(self, dout, cache):
        """Backward pass of Layer Normalization.

        Returns: (dz, dgamma, dbeta)
        """
        z_hat, std, gamma = cache
        h = z_hat.shape[1]
        # Your code here
        raise NotImplementedError

    def forward(self, X):
        """Forward pass. Cache all intermediates needed for backward.

        Returns: Y_hat of shape (B, C) — log-probabilities.
        """
        # Your code here
        # 1. Layer 1: Fully connected + LN + ReLU
        pass

        # 2. Layer 2: Dense connection [X || H1] + residual + LN + ReLU
        pass

        # 3. Layer 3: Dense connection [X || H1 || H2] + residual + LN + ReLU
        pass

        # 4. Output layer: Fully connected + stable log-softmax
        pass

        raise NotImplementedError

    def backward(self, X, y, lr=0.01):
        """Backpropagation and SGD update for all parameters."""
        batch_size = X.shape[0]
        y_onehot = np.eye(10)[y]

        # Your code here
        # 1. Output gradient: dZ4 = (softmax(Z4) - y_onehot) / B
        pass

        # 2. Grad for W4
        pass

        # 3. Backprop to H3 through ReLU
        pass

        # 4. Backprop through LN3
        pass

        # 5. Split residual at Layer 3: gradient goes to both Z3 and H2
        pass

        # 6. Grad for W3, then split C2 = [X || H1 || H2] gradient
        pass

        # 7. Accumulate dH2, backprop through ReLU2 and LN2
        pass

        # 8. Split residual at Layer 2, grad for W2, split C1 = [X || H1] gradient
        pass

        # 9. Accumulate dH1 (three paths: Layer 3 dense + Layer 2 dense + Layer 2 residual)
        pass

        # 10. Backprop through ReLU1 and LN1, grad for W1
        pass

        # 11. SGD update all parameters: W1-W4, gamma1-3, beta1-3
        pass

        raise NotImplementedError

    def predict(self, X):
        log_probs = self.forward(X)
        return np.argmax(log_probs, axis=1)


def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST DenseNet MLP with NumPy')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for SGD (default: 0.01)')
    parser.add_argument('--no-layernorm', action='store_true', default=False,
                        help='Disable Layer Normalization (you may need to reduce lr for stability)')
    args = parser.parse_args()

    # Load data
    X_train, X_test, y_train, y_test = load_fashion_mnist()

    # Initialize model
    use_layernorm = not args.no_layernorm
    model = NumpyDenseResMLP(input_size=784, hidden_size=args.hidden_size,
                             use_layernorm=use_layernorm)

    train_losses = []
    train_accs = []
    test_accs = []

    n_batches = len(X_train) // args.batch_size

    print(f"Total train size {len(X_train)}, Total test size {len(X_test)}")
    print(f"LayerNorm: {use_layernorm}, LR: {args.lr}, Hidden: {args.hidden_size}")

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0
        correct = 0
        total = 0

        for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{args.epochs}'):
            start_idx = i * args.batch_size
            end_idx = start_idx + args.batch_size

            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            log_probs = model.forward(X_batch)
            batch_loss = -np.mean(log_probs[np.arange(len(y_batch)), y_batch])
            epoch_loss += batch_loss
            model.backward(X_batch, y_batch, lr=args.lr)

            preds = np.argmax(log_probs, axis=1)
            correct += np.sum(preds == y_batch)
            total += len(y_batch)

        # Calculate epoch metrics
        train_loss = epoch_loss / n_batches
        train_acc = 100 * correct / total

        # Evaluate on test set
        test_preds = model.predict(X_test)
        test_acc = 100 * np.mean(test_preds == y_test)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Acc: {test_acc:.2f}%')

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('numpy_results.png')
    plt.show()


def main_learning_curve():
    # Your code here
    # Implement the training code that plots the learning curve.
    # You can use the save_learning_curve function from hw1_learning_curves.py.
    raise NotImplementedError


if __name__ == '__main__':
    main()

    # NOTE: Comment out the above main() and uncomment the below main_learning_curve()
    # to run the learning curve code for Q5.4
    # main_learning_curve()

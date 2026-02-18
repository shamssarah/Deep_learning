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

        self.use_layernorm = use_layernorm

        #Normal distribution of weights
        mean = 0.0
        std_dev = np.sqrt(2.0 / (input_size + 1))  
        self.W1 = np.random.normal(loc=mean, scale=std_dev, size=(input_size + 1, hidden_size) )

        std_dev = np.sqrt(2.0 / (input_size + hidden_size + 1))  
        self.W2 = np.random.normal(loc=mean, scale=std_dev, size=(input_size + hidden_size + 1, hidden_size))

        std_dev = np.sqrt(2.0 / (input_size + 2*hidden_size + 1))
        self.W3 = np.random.normal(loc=mean, scale=std_dev, size=(input_size + 2*hidden_size + 1, hidden_size))

        std_dev = np.sqrt(2.0 / (hidden_size + 1))
        self.W4 = np.random.normal(loc=mean, scale=std_dev, size=(hidden_size + 1, num_classes))

        self.gamma1 = np.ones(hidden_size)
        self.beta1 = np.zeros(hidden_size)
        self.gamma2 = np.ones(hidden_size)
        self.beta2 = np.zeros(hidden_size)
        self.gamma3 = np.ones(hidden_size)
        self.beta3 = np.zeros(hidden_size)

        # raise NotImplementedError

    def layer_norm(self, z, gamma, beta):
        """Forward pass of Layer Normalization (per sample, across hidden dim).

        Returns: (out, cache) where cache = (z_hat, std, gamma) for backward.
        """
        eps = 1e-5
        # Your code here


        mean = np.mean(z, axis=1, keepdims=True)
        var = np.var(z, axis=1, keepdims=True)
        std = np.sqrt(var + eps)
        z_hat = (z - mean) / std
        out = gamma * z_hat + beta

    
        return out, (z_hat, std, gamma)
        

    def layer_norm_backward(self, dout, cache):
        """Backward pass of Layer Normalization.

        Returns: (dz, dgamma, dbeta)
        """
        z_hat, std, gamma = cache
        h = z_hat.shape[1]
        # Your code here
        
        dgamma = np.sum(dout * z_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        d_zhat = dout * gamma
        dz = (1.0/(h*std)) * (h * d_zhat - np.sum(d_zhat, axis=1, keepdims=True) - z_hat * np.sum(d_zhat * z_hat, axis=1, keepdims=True))

        return dz, dgamma, dbeta
        # raise NotImplementedError

    def forward(self, X):
        """Forward pass. Cache all intermediates needed for backward.

        Returns: Y_hat of shape (B, C) — log-probabilities.
        """
        # Your code here
        # 1. Layer 1: Fully connected + LN + ReLU

        self.X_bar = append_ones(X)  
        Z1 = self.X_bar @ self.W1
        N1,self.cache1 = self.layer_norm (Z1,self.gamma1,self.beta1)
        self.H1 = np.maximum(0, N1)

        # pass

        # 2. Layer 2: Dense connection [X || H1] + residual + LN + ReLU
        
        self.C1 = np.hstack([X, self.H1])
        self.C1_bar = append_ones(self.C1)
        Z2 = self.C1_bar @ self.W2
        N2, self.cache2 = self.layer_norm(Z2 + self.H1, self.gamma2, self.beta2)
        self.H2 = np.maximum(0, N2)
        
        # pass

        # 3. Layer 3: Dense connection [X || H1 || H2] + residual + LN + ReLU
        
        self.C2 = np.hstack([X, self.H1, self.H2])
        self.C2_bar = append_ones(self.C2)
        Z3 = self.C2_bar @ self.W3
        N3, self.cache3 = self.layer_norm(Z3 + self.H2, self.gamma3, self.beta3)
        self.H3 = np.maximum(0, N3)

        # pass

        # 4. Output layer: Fully connected + stable log-softmax
        # pass

        self.H3_bar = append_ones(self.H3)
        Z4 = self.H3_bar @ self.W4
        Z4_max = np.max(Z4, axis=1, keepdims=True)
        Y_hat = Z4 - Z4_max - np.log(np.sum(np.exp(Z4-Z4_max),axis=1, keepdims=True)) #stablizing
        self.Y_hat = Y_hat
        return Y_hat

        # raise NotImplementedError

    def backward(self, X, y, lr=0.01):
        """Backpropagation and SGD update for all parameters."""
        batch_size = X.shape[0]
        y_onehot = np.eye(10)[y]

        # Your code here
        # 1. Output gradient: dZ4 = (softmax(Z4) - y_onehot) / B

        dZ4 = ( np.exp(self.Y_hat) - y_onehot) / batch_size
        # pass

        # 2. Grad for W4
        dW4 = self.H3_bar.T @ dZ4 

        # pass

        # 3. Backprop to H3 through ReLU
        dH3_bar = dZ4 @ self.W4.T
        dH3 = dH3_bar[:, :-1]
        # pass

        # 4. Backprop through LN3

        dN3 = dH3 * (self.H3 > 0)

        dL3_input, dgamma3, dbeta3 = self.layer_norm_backward(dN3, self.cache3)
        # pass

        # 5. Split residual at Layer 3: gradient goes to both Z3 and H2

        dZ3 = dL3_input 
        dH2 =  dL3_input  
        # pass

        # 6. Grad for W3, then split C2 = [X || H1 || H2] gradient
        dW3 = self.C2_bar.T @ dZ3
        dC2_bar = dZ3 @ self.W3.T
        dC2 = dC2_bar[:, :-1]

        dC2_X = dC2[:, :X.shape[1]]
        dC2_H1 = dC2[:, X.shape[1]:X.shape[1]+self.H1.shape[1]]
        dC2_H2 = dC2[:, X.shape[1]+self.H1.shape[1]:]
        # pass

        # 7. Accumulate dH2, backprop through ReLU2 and LN2
    
        total_dH2 = dH2 + dC2_H2

        # pass

        # 8. Split residual at Layer 2, grad for W2, split C1 = [X || H1] gradient
        
        dN2 = total_dH2 * (self.H2 > 0)
        dL_input2, dgamma2, dbeta2 = self.layer_norm_backward(dN2, self.cache2)
        dZ2 = dL_input2
        dH1 = dL_input2

        dW2 = self.C1_bar.T @ dZ2
        dC1_bar = dZ2 @ self.W2.T
        dC1 = dC1_bar[:, :-1]


        #spliting:
        dC1_X = dC1[:, :X.shape[1]]
        dC1_H1 = dC1[:, X.shape[1]:]

        #  pass

        # 9. Accumulate dH1 (three paths: Layer 3 dense + Layer 2 dense + Layer 2 residual)
        
        total_dH1 = dH1 + dC2_H1 + dC1_H1
        
        # pass

        # 10. Backprop through ReLU1 and LN1, grad for W1

        dN1 = total_dH1 * (self.H1 > 0)
        dL_input1, dgamma1, dbeta1 = self.layer_norm_backward(dN1, self.cache1)
        
        dZ1 = dL_input1
        
        dW1 = self.X_bar.T @ dZ1



        # pass

        # 11. SGD update all parameters: W1-W4, gamma1-3, beta1-3
        # pass

        self.W4 -= lr * dW4
        self.W3 -= lr * dW3
        self.W2 -= lr * dW2
        self.W1 -= lr * dW1

        self.gamma3 -= lr * dgamma3
        self.beta3 -= lr * dbeta3

        self.gamma2 -= lr * dgamma2
        self.beta2 -= lr * dbeta2

        self.gamma1 -= lr * dgamma1
        self.beta1 -= lr * dbeta1
    
        loss = -np.sum(self.Y_hat[np.arange(batch_size),y])/ batch_size
        return loss

        # raise NotImplementedError

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

    X_train, X_test, y_train, y_test = load_fashion_mnist()

    hidden_size = [128,256,256]
    lr = [0.01,0.01,0.001]
    layer_norm = [True,True,False]

    train_sizes = list(range(5500,56001,5500))

    EPOCHS = 50
    batch_size = 128

    # early_stop will occur when the loss/accuracy does not improve for 5 consecutive epochs
    for config_idx in range(3):

        config_train_accs = []
        config_test_accs = []
        dataset_sizes = []

        for train_size in train_sizes:
            print ("Configuration{}: hidden_size={}, lr={}, layer_norm={}, training_size={}".format(
                            config_idx, hidden_size[config_idx], lr[config_idx], layer_norm[config_idx], train_size
                            ))

            model = NumpyDenseResMLP(input_size=784, hidden_size=hidden_size[config_idx],
                            use_layernorm=layer_norm[config_idx])
    
            # defining the dataset here so that we have the same dataset for each configuration and each training size
            indices = np.random.permutation(len(X_train))
            # for ease of understanding, we will takout the subset we will be training on.
            Current_X = X_train[indices][:train_size]
            Current_y = y_train[indices][:train_size]

            validation_size = int(train_size * 0.20)
            real_train_size = train_size - validation_size
            
            X_train_shuffled = Current_X[:real_train_size]
            y_train_shuffled = Current_y[:real_train_size]
            # taking out a subsection out of the training data to perform validation loss calculation for early stopping
            X_val = Current_X[real_train_size:]
            y_val = Current_y[real_train_size:]

            n_batches = len(X_train_shuffled) // batch_size # batchsize

            previous_val_loss = float('inf')
            early_stop_counter = 0
            best_test_acc = 0
            best_train_acc = 0

            for epoch in range(EPOCHS): # to make sure we trigger early stopping, we set a large number of epochs
                for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{EPOCHS}'):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size

                    X_batch = X_train_shuffled[start_idx:end_idx]
                    y_batch = y_train_shuffled[start_idx:end_idx]

                    log_probs = model.forward(X_batch)                    
                    model.backward(X_batch, y_batch, lr=lr[config_idx])

     
                # Evaluate on test set
                #EARLY STOPPING : STOPPING IF THERE IS NOT CHANGE IN LOSS FOR 5 EPOCH

                val_log_probs = model.forward(X_val)
                val_loss = -np.mean(val_log_probs[np.arange(len(y_val)), y_val])
                
            
                test_preds = model.predict(X_test)
                test_acc = 100 * np.mean(test_preds == y_test)

                train_preds = model.predict(X_train_shuffled)
                train_acc = 100 * np.mean(train_preds == y_train_shuffled)
                
                if val_loss < (previous_val_loss - 1e-2):
                    early_stop_counter = 0
                    previous_val_loss = val_loss
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                  
                else:
                    early_stop_counter +=1

                if early_stop_counter >= 5:
                    print(f'Early stopping at epoch {epoch+1} due to no improvement in loss.')
                    break 
            

                print(f'\nEpoch {epoch+1}:')
                # print(f'Train Acc: {train_acc:.2f}%')
                # print (f'Val Loss: {val_loss:.4f}')
                # print(f'Test Acc: {test_acc:.2f}%')
            config_train_accs.append(best_train_acc)
            config_test_accs.append(best_test_acc)
            dataset_sizes.append(train_size)

        # collect results for plotting for each epoch
        # not using save_learning_curve due to visualization purpose 
        plt.figure(figsize=(12, 5))
        plt.plot(dataset_sizes, config_train_accs, label='Train Accuracy', marker='o')
        plt.plot(dataset_sizes, config_test_accs, label='Test Accuracy', marker='o')
        
        plt.title(f'Learning Curve: Config {config_idx} (Hidden={hidden_size[config_idx]}, LR={lr[config_idx]}, LayerNorm={layer_norm[config_idx]})')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        filename = f'learning_curve_config_{config_idx}.png'
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.show()
    # raise NotImplementedError


if __name__ == '__main__':
    # main()

    # NOTE: Comment out the above main() and uncomment the below main_learning_curve()
    # to run the learning curve code for Q5.4
    main_learning_curve()

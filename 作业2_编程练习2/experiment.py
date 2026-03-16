import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1 Target function
# f(x) = 0.3x^3 - 0.8x + sin(2x)
def target_function(x):
    return 0.3 * x**3 - 0.8 * x + np.sin(2 * x)

# 2 Dataset
def build_dataset(n_train=300, n_test=600):

    x_train = np.linspace(-2, 2, n_train).reshape(-1,1).astype(np.float32)
    y_train = target_function(x_train).astype(np.float32)

    x_test = np.linspace(-2, 2, n_test).reshape(-1,1).astype(np.float32)
    y_test = target_function(x_test).astype(np.float32)

    return x_train, y_train, x_test, y_test


# 3 Two-layer ReLU network
class TwoLayerReLU(nn.Module):

    def __init__(self, hidden_dim=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)



# 4 Training
def train_model(model, x_train, y_train, epochs=5000, lr=5e-3):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(1, epochs+1):

        optimizer.zero_grad()

        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}, Loss = {loss.item():.8f}")

    return losses



# 5 Evaluation
@torch.no_grad()
def evaluate_model(model, x_test, y_test):

    model.eval()

    y_pred = model(x_test)

    mse = torch.mean((y_pred - y_test) ** 2).item()
    max_error = torch.max(torch.abs(y_pred - y_test)).item()

    return y_pred.numpy(), mse, max_error



# 6 Plot
def plot_results(x_train, y_train, x_test, y_test, y_pred, losses):

    plt.figure(figsize=(8,5))

    plt.plot(x_test, y_test, label="True Function")
    plt.plot(x_test, y_pred, '--', label="Network Prediction")
    plt.scatter(x_train, y_train, s=10, alpha=0.5, label="Training Samples")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Two-Layer ReLU Function Approximation")

    plt.legend()
    plt.tight_layout()

    plt.savefig("relu_fit_curve.png", dpi=200)
    plt.close()


    plt.figure(figsize=(8,5))

    plt.plot(losses)

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title("Training Loss Curve")

    plt.tight_layout()

    plt.savefig("relu_fit_loss.png", dpi=200)
    plt.close()



# 7 Main
def main():

    np.random.seed(42)
    torch.manual_seed(42)

    x_train_np, y_train_np, x_test_np, y_test_np = build_dataset()

    x_train = torch.tensor(x_train_np)
    y_train = torch.tensor(y_train_np)

    x_test = torch.tensor(x_test_np)
    y_test = torch.tensor(y_test_np)

    model = TwoLayerReLU(hidden_dim=100)

    losses = train_model(model, x_train, y_train)

    y_pred, mse, max_error = evaluate_model(model, x_test, y_test)

    print("\nResults")
    print(f"Test MSE = {mse:.8f}")
    print(f"Max Absolute Error = {max_error:.8f}")

    plot_results(x_train_np, y_train_np, x_test_np, y_test_np, y_pred, losses)

    print("\nGenerated files:")
    print("relu_fit_curve.png")
    print("relu_fit_loss.png")


if __name__ == "__main__":
    main()
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_grad(x):
    t = np.tanh(x)
    return 1 - t**2


def forward(X, params, activation=np.tanh):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1
    a1 = activation(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    cache = (X, z1, a1, z2)
    return y_hat, cache


def loss_and_grads(X, y, params, activation=np.tanh, activation_grad=tanh_grad):
    n = len(X)
    W1, b1, W2, b2 = params
    y_hat, (X, z1, a1, z2) = forward(X, params, activation=activation)

    eps = 1e-9
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    delta2 = (y_hat - y) / n
    grad_W2 = a1.T @ delta2
    grad_b2 = np.sum(delta2, axis=0)

    delta1 = (delta2 @ W2.T) * activation_grad(z1)
    grad_W1 = X.T @ delta1
    grad_b1 = np.sum(delta1, axis=0)

    grads = (grad_W1, grad_b1, grad_W2, grad_b2)
    return loss, grads, y_hat


def update(params, grads, lr=0.1, weight_decay=0.0):
    W1, b1, W2, b2 = params
    gW1, gb1, gW2, gb2 = grads
    W1 = W1 - lr * (gW1 + weight_decay * W1)
    b1 = b1 - lr * gb1
    W2 = W2 - lr * (gW2 + weight_decay * W2)
    b2 = b2 - lr * gb2
    return (W1, b1, W2, b2)


def make_loss_plot(losses):
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="#4C72B0", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Binary cross-entropy")
    plt.title("Training loss on XOR (2-layer MLP)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "backprop_loss.png", dpi=150)
    plt.close()


def make_boundary_plot(X, y, params):
    W1, b1, W2, b2 = params
    grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    y_hat, _ = forward(grid, params)
    Z = y_hat.reshape(grid_x.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(grid_x, grid_y, Z, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.8)
    plt.colorbar(label="P(y=1)")
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k", s=80)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.title("Decision boundary learned by 2-layer MLP on XOR")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "backprop_boundary.png", dpi=150)
    plt.close()


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    rng = np.random.default_rng(0)
    h = 4
    W1 = rng.normal(scale=np.sqrt(1 / X.shape[1]), size=(2, h))
    b1 = np.zeros(h)
    W2 = rng.normal(scale=np.sqrt(1 / h), size=(h, 1))
    b2 = np.zeros(1)
    params = (W1, b1, W2, b2)

    losses = []
    for _ in range(3000):
        loss, grads, y_hat = loss_and_grads(X, y, params)
        params = update(params, grads, lr=0.15, weight_decay=1e-3)
        losses.append(loss)

    make_loss_plot(losses)
    make_boundary_plot(X, y, params)

    print("Final loss:", losses[-1])


if __name__ == "__main__":
    main()

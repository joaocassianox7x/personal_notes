import pathlib

import matplotlib.pyplot as plt
import numpy as np

# Paths
ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

# Synthetic data
n = 120
X = rng.uniform(-3, 3, size=(n, 1))
noise = rng.normal(scale=1.0, size=n)
y_true = 1.5 + 2.2 * X[:, 0]
y = y_true + noise

# Helpers

def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), x]


def fit_ols(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    x_ = add_intercept(x)
    beta = np.linalg.inv(x_.T @ x_) @ x_.T @ target
    return beta


def fit_ridge(x: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    x_ = add_intercept(x)
    n_features = x_.shape[1]
    beta = np.linalg.inv(x_.T @ x_ + lam * np.eye(n_features)) @ x_.T @ target
    return beta


def predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# Fit models
beta_ols = fit_ols(X, y)
beta_ridge = fit_ridge(X, y, lam=2.0)
y_hat_ols = predict(X, beta_ols)
y_hat_ridge = predict(X, beta_ridge)

# 1) Fit visualization
x_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)[:, None]
true_line = 1.5 + 2.2 * x_plot[:, 0]
ols_line = predict(x_plot, beta_ols)
ridge_line = predict(x_plot, beta_ridge)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y, color="#4C72B0", alpha=0.65, label="Data")
plt.plot(x_plot[:, 0], true_line, "k--", linewidth=2, label="True relation")
plt.plot(x_plot[:, 0], ols_line, color="#55A868", linewidth=2, label="OLS fit")
plt.plot(x_plot[:, 0], ridge_line, color="#C44E52", linewidth=2, label="Ridge (lambda=2)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression: true vs. fitted")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "linear_regression_fit.png", dpi=150)
plt.close()

# 2) Residual plot
residuals = y - y_hat_ols
plt.figure(figsize=(8, 5))
plt.scatter(y_hat_ols, residuals, color="#8172B3", alpha=0.7)
plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
plt.xlabel("Predicted (OLS)")
plt.ylabel("Residuals")
plt.title("Residuals vs. predictions")
plt.tight_layout()
plt.savefig(IMG_DIR / "linear_regression_residuals.png", dpi=150)
plt.close()

# 3) Ridge path (coefficient shrinkage)
lambdas = np.logspace(-3, 2, 40)
coefs = []
for lam in lambdas:
    coefs.append(fit_ridge(X, y, lam=lam))
coefs = np.array(coefs)

plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, coefs[:, 0], label="Intercept", color="#4C72B0")
plt.semilogx(lambdas, coefs[:, 1], label="Slope", color="#55A868")
plt.xlabel("lambda")
plt.ylabel("Coefficient value")
plt.title("Ridge coefficient shrinkage")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "linear_regression_ridge_path.png", dpi=150)
plt.close()

# Report metrics to stdout for quick sanity check
print("OLS beta:", beta_ols)
print("Ridge beta (lambda=2):", beta_ridge)
print("R^2 (OLS):", r2_score(y, y_hat_ols))

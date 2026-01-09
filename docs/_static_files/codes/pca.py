"""Generate PCA visualizations for documentation."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def compute_pca(X, n_components=None):
    """Compute PCA via SVD; return mean, components, singular values, explained variance, ratio."""
    if n_components is None:
        n_components = X.shape[1]
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    total_var = explained_variance.sum()
    explained_ratio = explained_variance / total_var
    return Xc, components, explained_variance, explained_ratio


def make_data(seed=7, n=400):
    rng = np.random.default_rng(seed)
    # Correlated 4D data with a strong 2D subspace
    cov = np.array(
        [
            [2.0, 1.4, 0.6, 0.4],
            [1.4, 1.8, 0.5, 0.3],
            [0.6, 0.5, 1.0, 0.2],
            [0.4, 0.3, 0.2, 0.6],
        ]
    )
    mean = np.array([0.0, 0.5, -0.2, 1.0])
    X = rng.multivariate_normal(mean, cov, size=n)
    return X


def plot_scree(explained_var, explained_ratio):
    idx = np.arange(1, len(explained_var) + 1)
    plt.figure(figsize=(6, 4))
    plt.bar(idx, explained_var, color="#4C72B0", alpha=0.8, label="Eigenvalues")
    plt.plot(idx, explained_var, color="#4C72B0", linewidth=2)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance (eigenvalue)")
    plt.title("Scree plot")
    plt.xticks(idx)
    plt.grid(True, axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = IMG_DIR / "pca_variance_explained.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")

    # cumulative
    plt.figure(figsize=(6, 4))
    plt.plot(idx, np.cumsum(explained_ratio), marker="o", color="#55A868", linewidth=2.5)
    plt.axhline(0.95, color="gray", linestyle=":", linewidth=1.2, label="95% threshold")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance ratio")
    plt.ylim(0, 1.05)
    plt.xticks(idx)
    plt.title("Cumulative explained variance")
    plt.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = IMG_DIR / "pca_cumulative_variance.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_projection(Xc, components):
    Z2 = Xc @ components[:2].T
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=Z2[:, 0], cmap="viridis", s=28, alpha=0.8, edgecolor="k", linewidth=0.2)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D projection on first two PCs")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = IMG_DIR / "pca_2d_projection.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_biplot(Xc, components, explained_ratio, feature_names=None):
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(components.shape[1])]
    Z2 = Xc @ components[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z2[:, 0], Z2[:, 1], c="#4C72B0", s=24, alpha=0.65, label="Samples")

    scale = 2.0
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, scale * components[0, i], scale * components[1, i],
                 head_width=0.05, head_length=0.08, fc="#DD8452", ec="#DD8452", linewidth=2)
        ax.text(scale * components[0, i] * 1.1, scale * components[1, i] * 1.1,
                name, color="#DD8452", fontsize=10, fontweight="bold")

    ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}% var)")
    ax.set_title("Biplot: samples and loadings")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = IMG_DIR / "pca_biplot.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✓ Saved: {out}")


def main():
    X = make_data()
    Xc, comps, var, ratio = compute_pca(X)
    plot_scree(var, ratio)
    plot_projection(Xc, comps)
    plot_biplot(Xc, comps, ratio)


if __name__ == "__main__":
    main()

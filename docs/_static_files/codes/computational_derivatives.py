"""Generate finite-difference figures for the computational derivatives section."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h


def backward_diff(f, x, h):
    return (f(x) - f(x - h)) / h


def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def plot_error_vs_h():
    x0 = 1.0
    true_deriv = np.cos(x0)
    hs = np.logspace(-8, -0.5, 200)
    f = np.sin

    forward_err = np.abs(forward_diff(f, x0, hs) - true_deriv)
    backward_err = np.abs(backward_diff(f, x0, hs) - true_deriv)
    central_err = np.abs(central_diff(f, x0, hs) - true_deriv)

    plt.figure(figsize=(6, 4))
    plt.loglog(hs, forward_err, label="Forward (O(h))")
    plt.loglog(hs, backward_err, label="Backward (O(h))", linestyle="--")
    plt.loglog(hs, central_err, label="Central (O(h^2))")
    plt.axvline(1e-4, color="gray", linestyle=":", linewidth=1)
    plt.axhline(1e-10, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Step size h")
    plt.ylabel("Absolute error at x=1")
    plt.title("Finite difference error vs h")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    out = IMG_DIR / "finite_difference_error.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_tradeoff():
    hs = np.logspace(-8, -0.5, 200)
    mach_eps = np.finfo(float).eps
    c_t = 1.0
    c_r = 1.0

    err_trunc = c_t * hs ** 2
    err_round = c_r * mach_eps / hs
    err_total = err_trunc + err_round
    h_star = (mach_eps / (2 * c_t)) ** (1 / 3)

    plt.figure(figsize=(6, 4))
    plt.loglog(hs, err_trunc, label="Truncation ~ h^2")
    plt.loglog(hs, err_round, label="Roundoff ~ eps/h")
    plt.loglog(hs, err_total, label="Total error", color="k")
    plt.axvline(h_star, color="red", linestyle=":", label=f"h* ~ {h_star:.1e}")
    plt.xlabel("Step size h")
    plt.ylabel("Error model (scaled)")
    plt.title("Truncation vs roundoff tradeoff")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    out = IMG_DIR / "finite_difference_tradeoff.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_stencil():
    fig, ax = plt.subplots(figsize=(6, 2))
    x_positions = [-1, 0, 1]
    labels = ["x-h", "x", "x+h"]
    colors = ["#4C72B0", "#55A868", "#4C72B0"]
    for x, lbl, c in zip(x_positions, labels, colors):
        rect = patches.FancyBboxPatch(
            (x - 0.45, -0.4),
            0.9,
            0.8,
            boxstyle="round,pad=0.05",
            linewidth=1,
            edgecolor=c,
            facecolor=c,
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(x, 0, lbl, ha="center", va="center", color="white", fontsize=11)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    ax.set_title("Central stencil uses symmetric points")
    plt.tight_layout()
    out = IMG_DIR / "finite_difference_stencil.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✓ Saved: {out}")


def main():
    plot_error_vs_h()
    plot_tradeoff()
    plot_stencil()


if __name__ == "__main__":
    main()

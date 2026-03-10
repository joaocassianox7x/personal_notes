"""Generate the Monte Carlo animation used in the OpenMP notes."""

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

OUT = IMG_DIR / "openmp_monte_carlo.gif"


def build_frames():
    rng = np.random.default_rng(12)
    samples = 1800
    frame_count = 36

    x = rng.random(samples)
    y = rng.random(samples)
    inside = x * x + y * y <= 1.0
    curve_x = np.linspace(0.0, 1.0, 500)
    curve_y = np.sqrt(1.0 - curve_x**2)

    stops = np.linspace(40, samples, frame_count, dtype=int)
    frames = []

    for stop in stops:
        current_inside = inside[:stop]
        estimate = current_inside.mean()
        pi_estimate = 4.0 * estimate

        fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
        ax.fill_between(curve_x, 0.0, curve_y, color="#9fd3c7", alpha=0.45)
        ax.plot(curve_x, curve_y, color="#155e63", linewidth=2.4, label=r"$y = \sqrt{1 - x^2}$")
        ax.scatter(
            x[:stop][current_inside],
            y[:stop][current_inside],
            s=14,
            color="#0d766e",
            alpha=0.72,
            label="Accepted",
        )
        ax.scatter(
            x[:stop][~current_inside],
            y[:stop][~current_inside],
            s=14,
            color="#c86b41",
            alpha=0.5,
            label="Rejected",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Monte Carlo hit-or-miss integral")
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.legend(loc="lower left", fontsize=9, frameon=True)
        ax.text(
            0.03,
            0.97,
            f"samples = {stop}\nI ≈ {estimate:.5f}\nπ ≈ {pi_estimate:.5f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    return frames


def main():
    frames = build_frames()
    imageio.mimsave(OUT, frames, duration=0.12, loop=0)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

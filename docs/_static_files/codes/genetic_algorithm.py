"""Generate figures for the genetic algorithm notes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BOUNDS = (-4.0, 4.0)
BITS = 12
POP_SIZE = 36
GENERATIONS = 45
ELITE_COUNT = 2
TOURNAMENT_K = 3


def objective(x):
    """Multimodal objective used in the visual examples."""
    x = np.asarray(x)
    return 1.9 + np.sin(2.6 * x) + 0.45 * np.cos(6.2 * x) - 0.06 * (x - 1.1) ** 2


def tournament_select(rng, population, fitness, k=TOURNAMENT_K):
    idx = rng.integers(0, len(population), size=k)
    winner = idx[np.argmax(fitness[idx])]
    return population[winner]


def run_binary_ga(seed=7):
    rng = np.random.default_rng(seed)
    weights = 2 ** np.arange(BITS - 1, -1, -1)

    def decode(pop_bits):
        ints = pop_bits @ weights
        scale = (BOUNDS[1] - BOUNDS[0]) / (2**BITS - 1)
        return BOUNDS[0] + ints * scale

    population = rng.integers(0, 2, size=(POP_SIZE, BITS), endpoint=False)
    history_best = []
    history_mean = []
    initial_x = decode(population)
    snapshots = []

    for gen in range(GENERATIONS):
        x = decode(population)
        fitness = objective(x)
        history_best.append(float(fitness.max()))
        history_mean.append(float(fitness.mean()))
        if gen in (0, GENERATIONS - 1):
            snapshots.append((x.copy(), fitness.copy()))

        elite_idx = np.argsort(fitness)[-ELITE_COUNT:]
        next_population = [population[i].copy() for i in elite_idx]

        while len(next_population) < POP_SIZE:
            p1 = tournament_select(rng, population, fitness).copy()
            p2 = tournament_select(rng, population, fitness).copy()

            if rng.random() < 0.9:
                cut = rng.integers(1, BITS)
                c1 = np.concatenate([p1[:cut], p2[cut:]])
                c2 = np.concatenate([p2[:cut], p1[cut:]])
            else:
                c1, c2 = p1, p2

            mutation_mask_1 = rng.random(BITS) < (1.0 / BITS)
            mutation_mask_2 = rng.random(BITS) < (1.0 / BITS)
            c1 = np.where(mutation_mask_1, 1 - c1, c1)
            c2 = np.where(mutation_mask_2, 1 - c2, c2)

            next_population.extend([c1, c2])

        population = np.array(next_population[:POP_SIZE], dtype=int)

    final_x = decode(population)
    final_fitness = objective(final_x)
    best_idx = int(np.argmax(final_fitness))

    return {
        "name": "Binary-coded GA",
        "initial_x": initial_x,
        "final_x": final_x,
        "history_best": np.array(history_best),
        "history_mean": np.array(history_mean),
        "best_x": float(final_x[best_idx]),
        "best_fitness": float(final_fitness[best_idx]),
        "snapshots": snapshots,
    }


def run_real_ga(seed=11):
    rng = np.random.default_rng(seed)
    low, high = BOUNDS
    population = rng.uniform(low, high, size=POP_SIZE)
    history_best = []
    history_mean = []
    initial_x = population.copy()
    snapshots = []

    for gen in range(GENERATIONS):
        fitness = objective(population)
        history_best.append(float(fitness.max()))
        history_mean.append(float(fitness.mean()))
        if gen in (0, GENERATIONS - 1):
            snapshots.append((population.copy(), fitness.copy()))

        elite_idx = np.argsort(fitness)[-ELITE_COUNT:]
        next_population = [population[i] for i in elite_idx]

        while len(next_population) < POP_SIZE:
            p1 = float(tournament_select(rng, population, fitness))
            p2 = float(tournament_select(rng, population, fitness))

            if rng.random() < 0.9:
                alpha = rng.random()
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = (1.0 - alpha) * p1 + alpha * p2
            else:
                c1, c2 = p1, p2

            c1 += rng.normal(0.0, 0.18)
            c2 += rng.normal(0.0, 0.18)
            c1 = float(np.clip(c1, low, high))
            c2 = float(np.clip(c2, low, high))
            next_population.extend([c1, c2])

        population = np.array(next_population[:POP_SIZE], dtype=float)

    final_fitness = objective(population)
    best_idx = int(np.argmax(final_fitness))

    return {
        "name": "Real-coded GA",
        "initial_x": initial_x,
        "final_x": population.copy(),
        "history_best": np.array(history_best),
        "history_mean": np.array(history_mean),
        "best_x": float(population[best_idx]),
        "best_fitness": float(final_fitness[best_idx]),
        "snapshots": snapshots,
    }


def plot_landscape(binary_run, real_run):
    xs = np.linspace(BOUNDS[0], BOUNDS[1], 600)
    ys = objective(xs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    configs = [
        (axes[0], binary_run, "#4C72B0", "#DD8452"),
        (axes[1], real_run, "#55A868", "#C44E52"),
    ]

    for ax, run, initial_color, final_color in configs:
        ax.plot(xs, ys, color="#1f2933", linewidth=2.2, label="Objective")
        ax.scatter(
            run["initial_x"],
            objective(run["initial_x"]),
            color=initial_color,
            alpha=0.45,
            s=36,
            label="Generation 0",
        )
        ax.scatter(
            run["final_x"],
            objective(run["final_x"]),
            color=final_color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            s=42,
            label=f"Generation {GENERATIONS - 1}",
        )
        ax.axvline(run["best_x"], color=final_color, linestyle=":", linewidth=1.6)
        ax.set_title(run["name"])
        ax.set_xlabel("Candidate value x")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="lower left", fontsize=9)
        ax.text(
            0.03,
            0.96,
            f"Best x = {run['best_x']:.3f}\nBest fitness = {run['best_fitness']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

    axes[0].set_ylabel("Fitness")
    fig.suptitle("Population movement on a multimodal objective", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = IMG_DIR / "genetic_algorithm_landscape.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


def plot_convergence(binary_run, real_run):
    gens = np.arange(GENERATIONS)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(gens, binary_run["history_best"], color="#4C72B0", linewidth=2.4, label="Binary GA best")
    ax.plot(gens, binary_run["history_mean"], color="#4C72B0", linewidth=1.8, linestyle="--", label="Binary GA mean")
    ax.plot(gens, real_run["history_best"], color="#C44E52", linewidth=2.4, label="Real-coded GA best")
    ax.plot(gens, real_run["history_mean"], color="#C44E52", linewidth=1.8, linestyle="--", label="Real-coded GA mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Convergence of two genetic algorithm variants")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / "genetic_algorithm_convergence.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


def plot_operators():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.axis("off")
    ax.set_title("Binary-coded operators", fontweight="bold")
    mono = {"family": "monospace", "fontsize": 14}

    parent_a = "101101|011001"
    parent_b = "010011|110100"
    child = "101101110100"
    mutated = "101101010100"

    ax.text(0.05, 0.84, "Parent A", fontsize=11, fontweight="bold")
    ax.text(0.28, 0.84, parent_a, **mono)
    ax.text(0.05, 0.67, "Parent B", fontsize=11, fontweight="bold")
    ax.text(0.28, 0.67, parent_b, **mono)
    ax.text(0.05, 0.48, "One-point crossover", fontsize=11)
    ax.annotate("", xy=(0.56, 0.62), xytext=(0.56, 0.76), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.annotate("", xy=(0.56, 0.62), xytext=(0.56, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(0.05, 0.30, "Offspring", fontsize=11, fontweight="bold")
    ax.text(0.28, 0.30, "101101110100", **mono)
    ax.text(0.05, 0.13, "Bit-flip mutation", fontsize=11)
    ax.text(0.28, 0.13, mutated, **mono)
    ax.text(0.51, 0.13, "^", color="#C44E52", **mono)
    ax.text(
        0.05,
        -0.02,
        "Discrete encoding: crossover swaps gene blocks, mutation flips bits.\n"
        "Useful when the search space is naturally symbolic or combinatorial.",
        fontsize=10,
    )

    ax = axes[1]
    ax.set_title("Real-coded operators", fontweight="bold")
    ax.set_xlim(-0.2, 5.4)
    ax.set_ylim(-0.2, 1.25)
    ax.axis("off")
    ax.hlines(0.35, 0, 5, color="#4b5563", linewidth=1.6)
    for tick in np.arange(0, 5.1, 1.0):
        ax.vlines(tick, 0.3, 0.4, color="#4b5563", linewidth=1)
        ax.text(tick, 0.2, f"{tick:.0f}", ha="center", fontsize=9)

    p1, p2 = 1.2, 3.8
    alpha = 0.35
    child = alpha * p1 + (1 - alpha) * p2
    mutated = child + 0.35

    ax.scatter([p1, p2], [0.35, 0.35], s=120, color="#4C72B0", zorder=3)
    ax.text(p1, 0.53, "Parent 1", ha="center", fontsize=10)
    ax.text(p2, 0.53, "Parent 2", ha="center", fontsize=10)
    ax.scatter([child], [0.35], s=140, color="#55A868", zorder=4)
    ax.text(child, 0.78, r"Child = \alpha p_1 + (1-\alpha)p_2", ha="center", fontsize=10)
    ax.annotate("", xy=(child, 0.42), xytext=(child, 0.68), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.scatter([mutated], [0.35], s=140, color="#C44E52", zorder=4)
    ax.annotate("", xy=(mutated, 0.35), xytext=(child, 0.35), arrowprops={"arrowstyle": "->", "lw": 2.0})
    ax.text(mutated, 0.53, r"Gaussian mutation", ha="center", fontsize=10)
    ax.text(
        0.0,
        -0.02,
        "Continuous encoding: arithmetic crossover interpolates between parents,\n"
        "then Gaussian mutation perturbs the child by a small random step.",
        fontsize=10,
    )

    fig.tight_layout()
    out = IMG_DIR / "genetic_algorithm_operators.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    binary_run = run_binary_ga()
    real_run = run_real_ga()
    plot_landscape(binary_run, real_run)
    plot_convergence(binary_run, real_run)
    plot_operators()
    print(
        "Binary-coded GA best:",
        f"x={binary_run['best_x']:.4f}, fitness={binary_run['best_fitness']:.4f}",
    )
    print(
        "Real-coded GA best:",
        f"x={real_run['best_x']:.4f}, fitness={real_run['best_fitness']:.4f}",
    )


if __name__ == "__main__":
    main()

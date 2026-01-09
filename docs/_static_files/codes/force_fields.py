"""Generate figures for force fields (water models) documentation."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT.parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


# Water model parameters: (name, sigma, epsilon, dipole, density)
MODELS = {
    "SPC": {"sigma": 3.1656, "epsilon": 0.1554, "dipole": 2.27, "density": 0.970},
    "SPC/E": {"sigma": 3.1656, "epsilon": 0.1554, "dipole": 2.35, "density": 0.995},
    "TIP3P": {"sigma": 3.1506, "epsilon": 0.1520, "dipole": 2.35, "density": 0.976},
    "TIP4P": {"sigma": 3.1536, "epsilon": 0.1550, "dipole": 2.17, "density": 0.998},
    "TIP4P/2005": {"sigma": 3.1589, "epsilon": 0.1852, "dipole": 2.305, "density": 0.994},
    "TIP4P/Ice": {"sigma": 3.1668, "epsilon": 0.1852, "dipole": 2.42, "density": 0.992},
    "TIP4P/Ew": {"sigma": 3.1640, "epsilon": 0.1852, "dipole": 2.08, "density": 0.994},
    "SPC/Fw": {"sigma": 3.1656, "epsilon": 0.1554, "dipole": 2.27, "density": 0.972},
}


def plot_lennard_jones_potential():
    """Plot LJ potential showing repulsion vs attraction."""
    r = np.linspace(2, 8, 500)
    sigma = 3.15
    eps = 0.155

    vdw = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    rep = 4 * eps * (sigma / r) ** 12
    att = -4 * eps * (sigma / r) ** 6

    plt.figure(figsize=(7, 5))
    plt.plot(r, vdw, "k-", linewidth=3, label="LJ 12-6 (Total)")
    plt.plot(r, rep, "r--", linewidth=2.5, label="Repulsive term (1/r¹²)")
    plt.plot(r, att, "b--", linewidth=2.5, label="Attractive term (-1/r⁶)")
    
    r_min = 2 ** (1/6) * sigma
    plt.axvline(r_min, color="green", linestyle=":", linewidth=2, label=f"r_min ≈ {r_min:.2f} Å")
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    
    plt.xlabel("Distance r (Å)", fontsize=11, fontweight="bold")
    plt.ylabel("Energy (kcal/mol)", fontsize=11, fontweight="bold")
    plt.title("Lennard-Jones 12-6 Potential: Repulsion vs Attraction", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.xlim(2, 8)
    plt.ylim(-0.3, 0.4)
    plt.tight_layout()
    out = IMG_DIR / "lennard_jones_potential.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_coulomb_vs_vdw():
    """Plot comparison of Coulomb and VdW interactions."""
    r = np.linspace(2, 10, 500)
    
    # Coulomb (arbitrary scale for visualization)
    coulomb = 1.0 / r
    
    # LJ
    sigma = 3.15
    eps = 0.155
    lj = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    
    # Dispersion only (attractive)
    dispersion = -4 * eps * (sigma / r) ** 6
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Full range
    ax1.plot(r, coulomb, "b-", linewidth=2.5, label="Coulomb ~ 1/r")
    ax1.plot(r, dispersion, "r--", linewidth=2.5, label="Dispersion ~ -1/r⁶")
    ax1.set_xlabel("Distance r (Å)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Energy (scaled)", fontsize=10, fontweight="bold")
    ax1.set_title("Long-range: Coulomb decays slower", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.set_xlim(2, 10)
    
    # Right: Close-range (LJ dominates)
    r_close = np.linspace(2.8, 5, 200)
    coulomb_close = 1.0 / r_close
    lj_close = 4 * eps * ((sigma / r_close) ** 12 - (sigma / r_close) ** 6)
    repulsion_close = 4 * eps * (sigma / r_close) ** 12
    
    ax2.plot(r_close, coulomb_close, "b-", linewidth=2.5, label="Coulomb")
    ax2.plot(r_close, lj_close, "k-", linewidth=2.5, label="LJ 12-6")
    ax2.plot(r_close, repulsion_close, "r--", linewidth=2, label="Repulsion (r⁻¹²)")
    ax2.set_xlabel("Distance r (Å)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Energy (scaled)", fontsize=10, fontweight="bold")
    ax2.set_title("Close-range: Repulsion dominates", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_xlim(2.8, 5)
    
    plt.tight_layout()
    out = IMG_DIR / "coulomb_vs_vdw.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_water_geometry():
    """Plot geometry of different water models."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    models_to_plot = [
        ("3-site (TIP3P, SPC)", [
            ([0, 0.9572*np.cos(52.26*np.pi/180)], [0, 0.9572*np.sin(52.26*np.pi/180)]),
            ([0, 0.9572*np.cos(-52.26*np.pi/180)], [0, 0.9572*np.sin(-52.26*np.pi/180)])
        ], "O", "H", None),
        ("4-site (TIP4P)", [
            ([0, 0.9572*np.cos(52.26*np.pi/180)], [0, 0.9572*np.sin(52.26*np.pi/180)]),
            ([0, 0.9572*np.cos(-52.26*np.pi/180)], [0, 0.9572*np.sin(-52.26*np.pi/180)])
        ], "O", "H", "M"),
        ("5-site (Drude)", [
            ([0, 0.9572*np.cos(52.26*np.pi/180)], [0, 0.9572*np.sin(52.26*np.pi/180)]),
            ([0, 0.9572*np.cos(-52.26*np.pi/180)], [0, 0.9572*np.sin(-52.26*np.pi/180)])
        ], "O", "H", "Drude"),
        ("Virtual site (TIP4P-like)", [
            ([0, 0.9572*np.cos(52.26*np.pi/180)], [0, 0.9572*np.sin(52.26*np.pi/180)]),
            ([0, 0.9572*np.cos(-52.26*np.pi/180)], [0, 0.9572*np.sin(-52.26*np.pi/180)])
        ], "O", "H", "Virtual M"),
    ]
    
    for ax, (title, bonds, o_label, h_label, extra_label) in zip(axes.flat, models_to_plot):
        # Oxygen atom (center)
        ax.scatter([0], [0], s=800, color="#E74C3C", edgecolors="black", linewidth=2, zorder=3)
        ax.text(0, -0.15, o_label, ha="center", fontsize=11, fontweight="bold")
        
        # Hydrogen atoms
        for i, bond in enumerate(bonds):
            x, y = bond[0][-1], bond[1][-1]
            ax.plot(bond[0], bond[1], "k-", linewidth=2)
            ax.scatter([x], [y], s=400, color="#3498DB", edgecolors="black", linewidth=1.5, zorder=3)
            offset_x, offset_y = 0.15*np.cos(np.arctan2(y, x)), 0.15*np.sin(np.arctan2(y, x))
            ax.text(x+offset_x, y+offset_y, h_label, ha="center", fontsize=10, fontweight="bold")
        
        # Extra site if present
        if extra_label:
            m_x, m_y = 0.08*np.cos(0), 0.08*np.sin(0)
            ax.scatter([m_x], [m_y], s=200, color="#F39C12", edgecolors="black", linewidth=1, zorder=3, marker="x")
            ax.text(m_x, m_y-0.15, extra_label, ha="center", fontsize=9, style="italic", color="#F39C12")
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    
    plt.tight_layout()
    out = IMG_DIR / "water_geometry.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out}")


def plot_model_comparison_bars():
    """Plot comparison of dipole moment and density across models."""
    names = list(MODELS.keys())
    dipoles = [MODELS[m]["dipole"] for m in names]
    densities = [MODELS[m]["density"] for m in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dipole moment
    colors_dip = ["#E74C3C" if abs(d - 1.85) > 0.3 else "#27AE60" for d in dipoles]
    bars1 = ax1.bar(range(len(names)), dipoles, color=colors_dip, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax1.axhline(1.85, color="blue", linestyle="--", linewidth=2, label="Experimental (1.85 D)")
    ax1.set_ylabel("Dipole Moment (Debye)", fontsize=11, fontweight="bold")
    ax1.set_title("Dipole Moment Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax1.legend(fontsize=10)
    ax1.set_ylim(1.5, 2.6)
    
    # Density
    colors_den = ["#E74C3C" if abs(d - 1.0) > 0.02 else "#27AE60" for d in densities]
    bars2 = ax2.bar(range(len(names)), densities, color=colors_den, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax2.axhline(1.0, color="blue", linestyle="--", linewidth=2, label="Experimental (1.0 g/cm³)")
    ax2.set_ylabel("Density (g/cm³)", fontsize=11, fontweight="bold")
    ax2.set_title("Bulk Density at 298 K", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.95, 1.01)
    
    plt.tight_layout()
    out = IMG_DIR / "water_models_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_rdf_example():
    """Plot radial distribution function with hydrogen bond peak."""
    r = np.linspace(0, 10, 300)
    
    # Simulated RDF with two peaks (H-bonds and second shell)
    rdf = 1.0 + 2.5 * np.exp(-((r - 2.8)**2) / 0.3) + 0.8 * np.exp(-((r - 4.5)**2) / 0.5) - 0.3 * np.exp(-((r - 6.0)**2) / 0.5)
    rdf = np.maximum(rdf, 0)
    
    plt.figure(figsize=(8, 5))
    plt.plot(r, rdf, "k-", linewidth=2.5)
    plt.fill_between(r, rdf, alpha=0.3)
    
    # Annotate peaks
    plt.axvline(2.8, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    plt.text(2.8, 3.5, "H-bond\npeak\n(~2.8 Å)", ha="center", fontsize=9, 
             bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    
    plt.axvline(4.5, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
    plt.text(4.5, 1.8, "Second shell\n(~4.5 Å)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    
    plt.xlabel("Distance r (Å)", fontsize=11, fontweight="bold")
    plt.ylabel("Radial Distribution Function g(r)", fontsize=11, fontweight="bold")
    plt.title("Oxygen-Oxygen Radial Distribution Function (RDF)", fontsize=12, fontweight="bold")
    plt.xlim(0, 10)
    plt.ylim(0, 4)
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = IMG_DIR / "water_rdf_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def plot_lennard_jones_parameters():
    """Plot LJ parameters (σ, ε) for different models."""
    names = list(MODELS.keys())
    sigmas = [MODELS[m]["sigma"] for m in names]
    epsilons = [MODELS[m]["epsilon"] for m in names]
    
    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    for name, sig, eps, color in zip(names, sigmas, epsilons, colors):
        plt.scatter(sig, eps, s=300, alpha=0.8, edgecolors="black", linewidth=1.5, color=color)
        plt.text(sig, eps + 0.002, name, ha="center", fontsize=8, fontweight="bold")
    
    plt.xlabel("σ (Å)", fontsize=11, fontweight="bold")
    plt.ylabel("ε (kcal/mol)", fontsize=11, fontweight="bold")
    plt.title("Lennard-Jones Parameters: σ vs ε for Water Models", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.xlim(3.14, 3.18)
    plt.ylim(0.145, 0.195)
    plt.tight_layout()
    out = IMG_DIR / "lennard_jones_parameters.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✓ Saved: {out}")


def main():
    plot_lennard_jones_potential()
    plot_coulomb_vs_vdw()
    plot_water_geometry()
    plot_model_comparison_bars()
    plot_rdf_example()
    plot_lennard_jones_parameters()


if __name__ == "__main__":
    main()

"""
PPPM vs. Ewald - Computational Comparison and Visualization

This script generates visualization comparing:
1. Computational complexity (time scaling)
2. Error vs. grid resolution (PPPM)
3. Error vs. Ewald parameter (Ewald)
4. Real vs. Reciprocal space contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erf
from scipy.fft import fftn, ifftn
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def coulomb_potential(r, alpha):
    """Coulomb potential decomposition."""
    erfc_part = erfc(alpha * r) / r
    erf_part = erf(alpha * r) / r
    return erfc_part, erf_part


def ewald_real_space_energy(positions, charges, alpha, cutoff, L):
    """
    Calculate real-space contribution to Ewald energy.
    
    Parameters:
    -----------
    positions : array, shape (N, 3)
        Particle positions
    charges : array, shape (N,)
        Particle charges
    alpha : float
        Ewald parameter
    cutoff : float
        Real-space cutoff radius
    L : float
        Box length (cubic)
    
    Returns:
    --------
    float
        Real-space energy
    """
    N = len(charges)
    energy = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            
            # Include periodic images
            for nx in [-1, 0, 1]:
                for ny in [-1, 0, 1]:
                    for nz in [-1, 0, 1]:
                        if nx == ny == nz == 0 and i == j:
                            continue
                        
                        image_shift = np.array([nx, ny, nz]) * L
                        r_image = np.linalg.norm(
                            positions[i] - positions[j] - image_shift
                        )
                        
                        if r_image < cutoff and r_image > 1e-10:
                            energy += 0.5 * charges[i] * charges[j] * \
                                     erfc(alpha * r_image) / r_image
    
    return energy


def ewald_self_energy(charges, alpha):
    """Self-interaction correction in Ewald."""
    return -(alpha / np.sqrt(np.pi)) * np.sum(charges**2)


def plot_complexity_comparison():
    """
    Plot computational complexity: Ewald vs PPPM
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # System sizes
    N = np.logspace(2, 5, 50)  # 100 to 100,000 particles
    
    # Complexity scalings
    ewald_complexity = N**(3/2)
    pppm_complexity = N * np.log(N)
    
    # Normalize for visibility
    ewald_normalized = ewald_complexity / ewald_complexity[0]
    pppm_normalized = pppm_complexity / pppm_complexity[0]
    
    ax.loglog(N, ewald_normalized, 'b-', linewidth=2.5, label=r'Ewald: $\mathcal{O}(N^{3/2})$')
    ax.loglog(N, pppm_normalized, 'r-', linewidth=2.5, label=r'PPPM: $\mathcal{O}(N \log N)$')
    
    # Highlight crossover region
    ax.axvspan(1000, 10000, alpha=0.1, color='green', label='PPPM advantage region')
    
    ax.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Computational Cost', fontsize=12, fontweight='bold')
    ax.set_title('Computational Complexity: Ewald vs PPPM', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'ewald_pppm_complexity.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: ewald_pppm_complexity.png")
    plt.close()


def plot_ewald_parameter_effect():
    """
    Plot how Ewald parameter alpha affects real vs reciprocal space contribution
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance range
    r = np.linspace(0.01, 5, 1000)
    alpha_values = [0.5, 1.0, 2.0, 5.0]
    
    # Left plot: Potential decomposition
    ax = axes[0]
    for alpha in alpha_values:
        erfc_contrib, erf_contrib = coulomb_potential(r, alpha)
        ax.semilogy(r, erfc_contrib, label=rf'$\alpha = {alpha}$ (real-space)', linewidth=2)
    
    ax.set_xlabel('Distance r', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'$\operatorname{erfc}(\alpha r) / r$', fontsize=11, fontweight='bold')
    ax.set_title('Real-Space Contribution vs Distance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-6, 1e1])
    
    # Right plot: Force magnitude
    ax = axes[1]
    for alpha in alpha_values:
        # Force ~ d/dr[erfc(alpha*r)/r]
        erfc_contrib, _ = coulomb_potential(r, alpha)
        force = -np.gradient(erfc_contrib, r)
        ax.semilogy(r, np.abs(force), label=rf'$\alpha = {alpha}$', linewidth=2)
    
    ax.set_xlabel('Distance r', fontsize=11, fontweight='bold')
    ax.set_ylabel('Force Magnitude', fontsize=11, fontweight='bold')
    ax.set_title('Real-Space Force vs Distance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'ewald_parameter_effect.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: ewald_parameter_effect.png")
    plt.close()


def plot_pppm_grid_convergence():
    """
    Plot PPPM accuracy vs grid resolution
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grid resolution (points per dimension)
    grid_sizes = np.array([4, 8, 16, 32, 64, 128])
    
    # Simplified convergence curves (realistic order of magnitude)
    # Error typically scales as grid_spacing^p where p depends on assignment order
    error_linear = 1.0 / grid_sizes  # Linear assignment
    error_quadratic = 1.0 / grid_sizes**2  # Quadratic assignment
    error_cubic = 1.0 / grid_sizes**4  # Cubic assignment
    
    # Left plot: Error vs grid resolution
    ax = axes[0]
    ax.loglog(grid_sizes, error_linear, 'o-', label='Linear assignment', linewidth=2, markersize=8)
    ax.loglog(grid_sizes, error_quadratic, 's-', label='Quadratic assignment', linewidth=2, markersize=8)
    ax.loglog(grid_sizes, error_cubic, '^-', label='Cubic assignment', linewidth=2, markersize=8)
    
    ax.set_xlabel('Grid Points per Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Error', fontsize=11, fontweight='bold')
    ax.set_title('PPPM: Convergence with Grid Resolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Right plot: Computational cost vs accuracy
    ax = axes[1]
    
    # Cost ~ grid_size^3 * log(grid_size)
    cost_linear = grid_sizes**3 * np.log(grid_sizes)
    
    # Normalize
    cost_linear_norm = cost_linear / cost_linear[-1]
    
    ax.loglog(grid_sizes, cost_linear_norm, 'g-', linewidth=2.5, label='FFT cost', marker='o', markersize=8)
    ax.loglog(grid_sizes, error_quadratic, 'b--', linewidth=2.5, label='Quadratic error', marker='s', markersize=8)
    
    ax.set_xlabel('Grid Points per Dimension', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Cost / Error', fontsize=11, fontweight='bold')
    ax.set_title('PPPM: Cost vs Accuracy Trade-off', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'pppm_grid_convergence.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: pppm_grid_convergence.png")
    plt.close()


def plot_energy_decomposition():
    """
    Plot energy contributions: real-space vs reciprocal-space for varying alpha
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    alpha_range = np.logspace(-1, 1, 50)
    
    # Create a simple test system: 2 opposite charges
    cutoff = 5.0
    r_ij = 1.0
    q1, q2 = 1.0, -1.0
    
    real_energies = []
    
    for alpha in alpha_range:
        # Real-space energy
        U_real = q1 * q2 * erfc(alpha * r_ij) / r_ij
        real_energies.append(abs(U_real))
    
    ax.semilogx(alpha_range, real_energies, 'b-', linewidth=2.5, label='Real-space contribution')
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'Typical $\alpha = 1.0$')
    
    ax.set_xlabel(r'Ewald Parameter $\alpha$', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Energy Contribution| (arb. units)', fontsize=12, fontweight='bold')
    ax.set_title(r'Ewald Energy: Real-Space Contribution vs $\alpha$ (r = 1.0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'energy_decomposition.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: energy_decomposition.png")
    plt.close()


def plot_method_selection_diagram():
    """
    Create a decision diagram for choosing between Ewald and PPPM
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create regions
    N_boundary = 10000
    accuracy_boundary = 1e-6
    
    # PPPM region (large N, moderate accuracy)
    pppm_x = np.array([N_boundary, 1e6, 1e6, N_boundary])
    pppm_y = np.array([1e-3, 1e-3, 1.0, 1.0])
    ax.fill(pppm_x, pppm_y, color='red', alpha=0.2, label='PPPM preferred')
    
    # Ewald region (small N, high accuracy)
    ewald_x = np.array([1e2, N_boundary, N_boundary, 1e2])
    ewald_y = np.array([1e-6, 1e-6, 1e-3, 1e-3])
    ax.fill(ewald_x, ewald_y, color='blue', alpha=0.2, label='Ewald preferred')
    
    # Transition region
    ax.fill([N_boundary, 1e5, 1e5, N_boundary], [1e-4, 1e-4, 1e-3, 1e-3], 
            color='yellow', alpha=0.2, label='Choice depends on needs')
    
    # Decision boundary
    ax.axvline(x=N_boundary, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(N_boundary, 0.8, f'N = {N_boundary}', fontsize=10, fontweight='bold', 
            ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acceptable Relative Error', fontsize=12, fontweight='bold')
    ax.set_title('Method Selection Guide: Ewald vs PPPM', fontsize=14, fontweight='bold')
    ax.set_xlim([1e2, 1e6])
    ax.set_ylim([1e-7, 1.0])
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'method_selection_diagram.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: method_selection_diagram.png")
    plt.close()


def plot_reciprocal_space_dampening():
    """
    Plot reciprocal-space contribution dampening effect
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # k-space vectors
    k = np.linspace(0.01, 10, 1000)
    alpha_values = [0.5, 1.0, 2.0, 5.0]
    
    for alpha in alpha_values:
        # Reciprocal space factor
        recip_factor = np.exp(-k**2 / (4 * alpha**2)) / k**2
        ax.semilogy(k, recip_factor, linewidth=2.5, label=rf'$\alpha = {alpha}$')
    
    ax.set_xlabel(r'Reciprocal-space vector magnitude k', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$e^{-k^2/(4\alpha^2)} / k^2$', fontsize=12, fontweight='bold')
    ax.set_title(r'Reciprocal-Space Dampening Effect', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_xlim([0, 10])
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'reciprocal_space_dampening.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: reciprocal_space_dampening.png")
    plt.close()


def main():
    """Generate all comparison plots."""
    print("\n" + "="*60)
    print("  PPPM vs Ewald - Generating Comparison Visualizations")
    print("="*60 + "\n")
    
    print(f"Output directory: {IMAGES_DIR}\n")
    
    print("Generating plots...")
    plot_complexity_comparison()
    plot_ewald_parameter_effect()
    plot_pppm_grid_convergence()
    plot_energy_decomposition()
    plot_method_selection_diagram()
    plot_reciprocal_space_dampening()
    
    print("\n" + "="*60)
    print("  ✓ All visualizations generated successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

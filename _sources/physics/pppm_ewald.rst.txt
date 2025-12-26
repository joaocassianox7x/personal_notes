PPPM vs. Ewald (Long-Range Coulomb)
===================================

Goal
----
Efficiently compute long-range electrostatics in periodic systems while keeping forces and energies accurate.

Classical Ewald summation
-------------------------
- Splits the potential into short-range (real-space) and long-range (reciprocal-space) sums using an Ewald parameter :math:`\alpha`.
- Real-space: rapidly decaying :math:`\operatorname{erfc}` term; cutoff in direct space.
- Reciprocal-space: Fourier sum over :math:`\mathbf{k}`-vectors.
- Cost: roughly :math:`\mathcal{O}(N^{3/2})`; accurate but expensive as :math:`N` grows.

PPPM (Particle–Particle Particle–Mesh)
--------------------------------------
- Also splits short/long-range but evaluates long-range on a mesh with FFTs.
- Steps: spread charges to a grid (assignment function), solve Poisson on the mesh via FFT, interpolate forces back to particles.
- Cost: :math:`\mathcal{O}(N \log N)` with controlled error via grid resolution and assignment order.
- Good for large systems; trades slight discretization error for speed.

Practical comparison
--------------------
- Ewald: simpler; tunable via :math:`\alpha` plus real/reciprocal cutoffs; best when :math:`N` is modest or very high accuracy is needed.
- PPPM: faster scaling; accuracy controlled by grid spacing, interpolation order, and real-space cutoff; ideal for large periodic simulations.
- Both rely on balancing the real-space and reciprocal/mesh errors through parameter choice.

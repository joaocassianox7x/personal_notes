:orphan:

PPPM vs. Ewald (Long-Range Coulomb)
===================================

Goal
Efficiently compute long-range electrostatics in periodic systems while keeping forces and energies accurate.

.. contents:: Contents
   :local:

Classical Ewald Summation
==========================

The Ewald method decomposes the Coulomb potential into short-range (real-space) and long-range (reciprocal-space) components:

Potential Decomposition
-----------------------

The total electrostatic potential is split as:

.. math::

    U_{\text{total}} = U_{\text{real}} + U_{\text{recip}} + U_{\text{self}}

**Step 1: Introduce the error function decomposition**

.. math::

    \frac{1}{r} = \frac{\operatorname{erfc}(\alpha r)}{r} + \frac{\operatorname{erf}(\alpha r)}{r}

where :math:`\alpha` is the Ewald parameter (controls the split between real and reciprocal space).

**Step 2: Real-space contribution**

The short-range part uses the complementary error function:

.. math::

    U_{\text{real}} = \frac{1}{2} \sum_{n \neq 0} \sum_{i,j} \frac{q_i q_j \operatorname{erfc}(\alpha r_{ij}^n)}{r_{ij}^n}

.. image:: ../_static_files/images/ewald_parameter_effect.png
   :alt: Real-space and force contributions for different Ewald parameters
   :width: 100%
   :align: center

where:

- :math:`r_{ij}^n = |\mathbf{r}_i - \mathbf{r}_j - \mathbf{L}n|` is the distance including periodic images (image index :math:`n`).
- The sum is truncated at a cutoff radius :math:`r_c`.
- This converges rapidly for larger :math:`\alpha`.

**Step 3: Reciprocal-space contribution**

The long-range part is evaluated in Fourier space:

.. math::

    U_{\text{recip}} = \frac{1}{2V} \sum_{\mathbf{k} \neq 0} \frac{4\pi}{k^2} e^{-k^2/(4\alpha^2)} \left| \sum_{i} q_i e^{i\mathbf{k} \cdot \mathbf{r}_i} \right|^2

.. image:: ../_static_files/images/reciprocal_space_dampening.png
   :alt: Reciprocal-space factor dampening for different Ewald parameters
   :width: 100%
   :align: center

where:

- :math:`\mathbf{k}` are reciprocal-space vectors: :math:`\mathbf{k} = 2\pi (n_x/L_x, n_y/L_y, n_z/L_z)` with :math:`n_x, n_y, n_z` integers.
- :math:`V` is the system volume.
- The sum converges exponentially due to the :math:`e^{-k^2/(4\alpha^2)}` factor.

**Step 4: Self-interaction correction**

.. math::

    U_{\text{self}} = -\frac{\alpha}{\sqrt{\pi}} \sum_i q_i^2

This removes the spurious self-energy from the continuous charge distribution approximation.

**Step 5: Total energy**

.. math::

    U_{\text{Ewald}} = U_{\text{real}} + U_{\text{recip}} + U_{\text{self}}

Computational Cost
-------------------

- Real-space: :math:`\mathcal{O}(N \cdot N_{\text{pairs}})` with short cutoff → effectively :math:`\mathcal{O}(N)`.
- Reciprocal-space: :math:`\mathcal{O}(N_{\mathbf{k}})` where :math:`N_{\mathbf{k}} \approx N` for comparable accuracy.
- **Total: approximately** :math:`\mathcal{O}(N^{3/2})` overall; expensive for large :math:`N`.

.. image:: ../_static_files/images/ewald_pppm_complexity.png
   :alt: Computational complexity comparison between Ewald and PPPM methods
   :width: 100%
   :align: center

*Ewald becomes computationally expensive for systems with more than ~10,000 particles.*

Forces
-------

Forces are obtained from the negative gradient:

.. math::

    \mathbf{F}_i = -\nabla_i U_{\text{Ewald}}

Real-space forces:

.. math::

    \mathbf{F}_{i, \text{real}} = q_i \sum_{j,n} q_j \frac{\mathbf{r}_{ij}^n}{(r_{ij}^n)^3} \left[ \operatorname{erfc}(\alpha r_{ij}^n) + \frac{2\alpha}{\sqrt{\pi}} r_{ij}^n e^{-(\alpha r_{ij}^n)^2} \right]

Reciprocal-space forces are computed via FFT derivatives (see PPPM section).

PPPM (Particle–Particle Particle–Mesh)
======================================

PPPM also splits the potential into short- and long-range parts but uses a mesh and FFT for efficiency.

Decomposition Strategy
----------------------

Like Ewald, we separate:

.. math::

    U_{\text{total}} = U_{\text{real}} + U_{\text{mesh}}

**Step 1: Real-space (short-range)**

Same as Ewald:

.. math::

    U_{\text{real}} = \frac{1}{2} \sum_{n \neq 0} \sum_{i,j} \frac{q_i q_j \operatorname{erfc}(\alpha r_{ij}^n)}{r_{ij}^n}

with cutoff :math:`r_c`.

Mesh-based Long-Range Calculation
----------------------------------

**Step 2: Assign charges to the mesh**

Charges are spread onto a regular 3D grid using an interpolation function :math:`w(x)`:

.. math::

    \rho(\mathbf{r}) = \sum_i q_i w(\mathbf{r} - \mathbf{r}_i)

Common choice: polynomial (e.g., linear, quadratic, cubic) assignment functions.

**Step 3: Solve Poisson's equation on the mesh**

In reciprocal space:

.. math::

    \hat{\phi}(\mathbf{k}) = \frac{4\pi}{k^2} e^{-k^2/(4\alpha^2)} \hat{\rho}(\mathbf{k})

where:

- :math:`\hat{\rho}` is the FFT of the gridded charge density.
- The factor :math:`e^{-k^2/(4\alpha^2)}` damps high-frequency modes (smoothing).

Compute inverse FFT to get the potential on the mesh:

.. math::

    \phi(\mathbf{r}) = \mathcal{F}^{-1}\left[ \hat{\phi}(\mathbf{k}) \right]

**Step 4: Interpolate forces back to particles**

The electric field on the mesh:

.. math::

    \mathbf{E}(\mathbf{r}) = -\nabla \phi(\mathbf{r})

Interpolate to particle positions using the same function :math:`w`:

.. math::

    \mathbf{E}_i = \sum_{\mathbf{r}_{\text{grid}}} \mathbf{E}(\mathbf{r}_{\text{grid}}) w(\mathbf{r}_{\text{grid}} - \mathbf{r}_i)

**Step 5: Mesh-based force**

.. math::

    \mathbf{F}_{i, \text{mesh}} = q_i \mathbf{E}_i

Computational Cost
-------------------

- Charge assignment: :math:`\mathcal{O}(N)`.
- FFT: :math:`\mathcal{O}(N_{\text{grid}} \log N_{\text{grid}})` where :math:`N_{\text{grid}}` is the number of grid points.
- Interpolation: :math:`\mathcal{O}(N)`.
- Real-space: :math:`\mathcal{O}(N)` with short cutoff.
- **Total: approximately** :math:`\mathcal{O}(N \log N)` — much faster than Ewald for large systems.

Accuracy Control
-----------------

Errors in PPPM arise from:

1. **Grid discretization error** — reduced by increasing grid resolution.
2. **Assignment/interpolation error** — reduced by using higher-order assignment functions.
3. **Real-space cutoff error** — reduced by increasing :math:`r_c` and/or :math:`\alpha`.

Fine-tuning these parameters allows direct control of accuracy versus speed.

.. image:: ../_static_files/images/pppm_grid_convergence.png
   :alt: PPPM convergence with grid resolution and accuracy-cost trade-offs
   :width: 100%
   :align: center

Practical Comparison
====================

.. image:: ../_static_files/images/method_selection_diagram.png
   :alt: Decision diagram for selecting between Ewald and PPPM methods
   :width: 100%
   :align: center

Ewald Summation
---------------

**Advantages:**

- Exact (within machine precision) for a given :math:`\alpha`.
- Straightforward parameter tuning via :math:`\alpha`, real/reciprocal cutoffs.
- Well-established for small to moderate systems.

**Disadvantages:**

- Computational scaling :math:`\mathcal{O}(N^{3/2})` becomes prohibitive for :math:`N > 10^4` particles.
- Requires careful selection of :math:`\alpha` and both real and reciprocal cutoffs.

**Best for:**

- Systems with modest particle count (:math:`N \lesssim 10^4`).
- Applications requiring high accuracy and exact Coulomb summation.

PPPM Method
-----------

**Advantages:**

- Fast scaling :math:`\mathcal{O}(N \log N)` suitable for large systems.
- Errors controlled by independent, tunable parameters (grid size, assignment order, real-space cutoff).
- Efficient FFT implementations on modern hardware.

**Disadvantages:**

- Introduces discretization error (not exact).
- Requires more careful parameter selection and validation.
- Grid memory overhead for very large cutoffs.

**Best for:**

- Large periodic simulations (:math:`N > 10^4` particles).
- High-throughput molecular dynamics and particle simulations.
- Systems where slight discretization errors are acceptable for speed gains.

Summary Table
=============

.. image:: ../_static_files/images/energy_decomposition.png
   :alt: Energy contribution from real-space component vs Ewald parameter
   :width: 100%
   :align: center

.. list-table:: Ewald vs. PPPM Comparison
   :widths: 25, 35, 35
   :header-rows: 1

   * - Aspect
     - Ewald
     - PPPM
   * - **Computational Complexity**
     - :math:`\mathcal{O}(N^{3/2})`
     - :math:`\mathcal{O}(N \log N)`
   * - **Accuracy**
     - Machine precision
     - Tunable via grid/assignment
   * - **Best System Size**
     - :math:`N \lesssim 10^4`
     - :math:`N \gg 10^4`
   * - **Primary Parameters**
     - :math:`\alpha`, :math:`r_c^{\text{real}}`, :math:`r_c^{\text{recip}}`
     - Grid size, assignment order, :math:`r_c`, :math:`\alpha`
   * - **Memory**
     - :math:`\mathcal{O}(N)`
     - :math:`\mathcal{O}(N_{\text{grid}})`

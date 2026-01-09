Computational Derivatives
=========================

Goal
----
Summarize common finite-difference formulas, emphasize the advantages of central differences, and discuss truncation error, roundoff error, and error propagation.

Finite differences (1D)
-----------------------
For a smooth function :math:`f(x)`, step :math:`h`:

- Forward: :math:`f'(x) \approx \dfrac{f(x+h)-f(x)}{h}` with truncation error :math:`\mathcal{O}(h)`.
- Backward: :math:`f'(x) \approx \dfrac{f(x)-f(x-h)}{h}` with truncation error :math:`\mathcal{O}(h)`.
- Central: :math:`f'(x) \approx \dfrac{f(x+h)-f(x-h)}{2h}` with truncation error :math:`\mathcal{O}(h^2)`.

Error order derivation (sketch)
-------------------------------
Taylor-expand around :math:`x`:

.. math::

    f(x \pm h) = f(x) \pm h f'(x) + \tfrac{h^2}{2} f''(x) \pm \tfrac{h^3}{6} f^{(3)}(x) + \mathcal{O}(h^4).

- Forward: subtract :math:`f(x)` and divide by :math:`h` :math:`\Rightarrow f'(x) + \tfrac{h}{2} f''(x) + \mathcal{O}(h^2)`. Leading error :math:`\propto h`.
- Backward: analogous, leading error :math:`-\tfrac{h}{2} f''(x)` :math:`\mathcal{O}(h)`.
- Central: subtract the backward from forward and divide by :math:`2h` :math:`\Rightarrow f'(x) + \tfrac{h^2}{6} f^{(3)}(x) + \mathcal{O}(h^4)`. Leading error :math:`\propto h^2`.

Visual intuition
----------------

.. image:: ../_static_files/images/finite_difference_stencil.png
   :alt: Central stencil symmetry
   :align: center
   :width: 70%

.. image:: ../_static_files/images/finite_difference_error.png
   :alt: Error versus step size for forward, backward, and central differences
   :align: center
   :width: 80%

Why central differences are better
----------------------------------
- **Higher accuracy:** symmetric cancellation yields second-order accuracy for the same step size :math:`h`.
- **Reduced phase error:** for oscillatory signals, central schemes preserve phase better than one-sided stencils.
- **Balanced bias:** forward/backward introduce directional bias; central is unbiased to leading order.

Second derivatives
------------------
.. math::

    f''(x) \approx \frac{f(x+h) - 2 f(x) + f(x-h)}{h^2} + \mathcal{O}(h^2).

Sketch: combine the Taylor expansions above; the :math:`h` terms cancel, leaving :math:`f''(x) + \tfrac{h^2}{12} f^{(4)}(x) + \mathcal{O}(h^4)`.

Error sources
-------------
- **Truncation error:** comes from omitting higher-order terms in the Taylor expansion. For central first derivative: :math:`E_T \approx \tfrac{h^2}{6} f^{(3)}(\xi)`.
- **Roundoff error:** subtractive cancellation when :math:`h` is very small; scales roughly like :math:`\varepsilon_{\text{mach}}/h`.
- **Total error tradeoff:** choose :math:`h` to balance truncation (:math:`\propto h^2`) and roundoff (:math:`\propto 1/h`).

.. image:: ../_static_files/images/finite_difference_tradeoff.png
    :alt: Truncation versus roundoff error model showing optimal h
    :align: center
    :width: 80%

Optimal step size (central, 2nd order)
--------------------------------------
Model total error magnitude as :math:`E(h) \approx C_T h^2 + C_R \varepsilon_{\text{mach}}/h`.
Setting :math:`\partial E / \partial h = 0` yields

.. math::

    h_\star = \left(\frac{C_R \varepsilon_{\text{mach}}}{2 C_T}\right)^{1/3} \approx (\varepsilon_{\text{mach}})^{1/3} \times \text{(problem scale)}.

In double precision (\(\varepsilon_{\text{mach}} \approx 2\times10^{-16}\)), :math:`h_\star` is typically :math:`\mathcal{O}(10^{-5} - 10^{-4})` times the scale of :math:`x`.
- **Error propagation:** in multi-step computations (e.g., time integration), local derivative error can amplify; monitor with refinement studies.

Practical guidance
------------------
1. Use central differences for interior points; fall back to one-sided stencils only at boundaries.
2. Pick :math:`h` relative to scale: :math:`h = \alpha \max(|x|, 1)\sqrt[3]{\varepsilon_{\text{mach}}}` with :math:`\alpha \in [10, 100]` often works in double precision.
3. For noisy data, avoid very small :math:`h`; denoise or fit a smooth surrogate first.
4. Validate with step-halving: compute derivative with :math:`h` and :math:`h/2`; second-order schemes should improve by factor :math:`\approx 4`.

NumPy snippet (central difference)
----------------------------------
.. code-block:: python

    import numpy as np

    def central_diff(f, x, h=None):
        # Choose h based on magnitude of x and machine epsilon
        if h is None:
            eps = np.finfo(float).eps
            h = (np.abs(x) + 1.0) * (eps ** (1/3))
        return (f(x + h) - f(x - h)) / (2 * h)

    # Example: derivative of sin at x=1.0
    val = central_diff(np.sin, 1.0)
    print(val)  # ~cos(1.0)

Extending to gradients
----------------------
For :math:`f: \mathbb{R}^n \to \mathbb{R}`, apply central differences per dimension:

.. math::

    \frac{\partial f}{\partial x_i}(\mathbf{x}) \approx \frac{f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x} - h \mathbf{e}_i)}{2h}.

Use the same :math:`h` balancing logic and beware correlated roundoff when dimensions are large.

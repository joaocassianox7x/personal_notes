Quantum Harmonic Oscillator
===========================

Setup and potential
-------------------
1D harmonic potential: :math:`V(x) = \tfrac{1}{2} m \omega^2 x^2`, a symmetric parabola centered at :math:`x=0`. Stationary Schr\"odinger equation

.. math::

   -\frac{\hbar^2}{2m} \frac{d^2 \psi}{dx^2} + \tfrac{1}{2} m \omega^2 x^2 \psi = E \psi.

Step-by-step wavefunction solution
----------------------------------
1. Non-dimensionalize with :math:`\xi = \sqrt{\tfrac{m\omega}{\hbar}}\,x` to simplify constants.
2. Rewrite the equation as :math:`\psi''(\xi) - \xi^2 \psi(\xi) + 2\epsilon\,\psi(\xi) = 0` with :math:`\epsilon = E/(\hbar\omega)`.
3. Impose normalizability by factoring :math:`\psi(\xi) = h(\xi) e^{-\xi^2/2}`; then :math:`h(\xi)` satisfies Hermite's equation.
4. Polynomial solutions require :math:`2\epsilon = 2n + 1`, giving energies :math:`E_n = \hbar \omega (n + 1/2)`.
5. Normalized eigenfunctions become

.. math::

   \psi_n(x) = \left(\frac{m\omega}{\pi \hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}}\, H_n(\xi)\, e^{-\xi^2/2}, \quad \xi = \sqrt{\tfrac{m\omega}{\hbar}} x.

Step-by-step ladder-operator (Dirac) solution
----------------------------------------------
1. Define

.. math::

   a = \sqrt{\tfrac{m\omega}{2\hbar}}\,x + \frac{i}{\sqrt{2m\hbar\omega}}\,p, \qquad
   a^\dagger = \sqrt{\tfrac{m\omega}{2\hbar}}\,x - \frac{i}{\sqrt{2m\hbar\omega}}\,p.

2. Verify commutator: :math:`[a,a^\dagger]=1` from :math:`[x,p]=i\hbar`.
3. Rewrite Hamiltonian: :math:`H = \hbar \omega (a^\dagger a + 1/2)`.
4. Define ground state by :math:`a|0\rangle = 0`; its energy is :math:`E_0 = \tfrac{1}{2}\hbar\omega`.
5. Excited states arise algebraically: :math:`|n\rangle = (a^\dagger)^n / \sqrt{n!} \, |0\rangle` with :math:`H|n\rangle = \hbar\omega(n+1/2)|n\rangle`.

Energy levels and potential
---------------------------
.. figure:: ../_static_files/images/energy_levels_harmonic.svg
   :alt: Harmonic oscillator potential with discrete energy levels
   :align: center
   :figwidth: 70%

   Quadratic potential with evenly spaced energy levels :math:`E_n = \hbar\omega(n+1/2)`.

Eigenfunctions (first six levels)
---------------------------------
.. figure:: ../_static_files/images/energy_eigenfunctions_harmonic.svg
   :alt: First eigenfunctions of the harmonic oscillator
   :align: center
   :figwidth: 80%

   Normalized eigenfunctions :math:`\psi_n(x)` for :math:`n=0..5`, illustrating nodes and Gaussian envelope.

Remarks
-------
- Wavefunctions are localized by the Gaussian factor; increasing :math:`n` adds nodes via :math:`H_n`.
- Equally spaced levels imply constant transition energy :math:`\hbar\omega`.
- In 3D, degeneracy grows with total quantum number; creation/annihilation operators generalize componentwise.

Split-Operator Method (Quantum Time Propagation)
==============================================

The split-operator (a.k.a. split-step Fourier) method is a workhorse algorithm to evolve the time-dependent Schr\"odinger equation

.. math::

   i\hbar\,\partial_t\psi(x,t) = \hat{H}\,\psi(x,t),\qquad \hat{H}=\hat{T}+\hat{V}.

In 1D, typically

.. math::

   \hat{T} = \frac{\hat{p}^2}{2m} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2},
   \qquad (\hat{V}\psi)(x)=V(x)\psi(x).

The math behind it (Trotter + Strang)
------------------------------------
The formal time-evolution over a step :math:`\Delta t` is

.. math::

   \psi(t+\Delta t) = \hat{U}(\Delta t)\,\psi(t),\qquad
   \hat{U}(\Delta t)=\exp\!\left(-\frac{i}{\hbar}\hat{H}\,\Delta t\right).

The difficulty is that :math:`[\hat{T},\hat{V}]\neq 0` in general, so

.. math::

   e^{-\frac{i}{\hbar}(\hat{T}+\hat{V})\Delta t}\neq e^{-\frac{i}{\hbar}\hat{T}\Delta t}\,e^{-\frac{i}{\hbar}\hat{V}\Delta t}.

A key idea is the **Trotter product formula**: for operators :math:`\hat{A},\hat{B}` (under appropriate conditions),

.. math::

   e^{\hat{A}+\hat{B}} = \lim_{n\to\infty}\left(e^{\hat{A}/n}e^{\hat{B}/n}\right)^n.

For finite :math:`\Delta t`, we use a higher-accuracy symmetric split (Strang splitting):

.. math::

   e^{(\hat{A}+\hat{B})\Delta t}
   = e^{\hat{A}\Delta t/2}\,e^{\hat{B}\Delta t}\,e^{\hat{A}\Delta t/2} + \mathcal{O}(\Delta t^3).

With

.. math::

   \hat{A}=-\frac{i}{\hbar}\hat{V},\qquad \hat{B}=-\frac{i}{\hbar}\hat{T},

we obtain the standard split-operator propagator

.. math::

   \hat{U}(\Delta t)
   \approx
   e^{-\frac{i}{\hbar}\hat{V}\,\Delta t/2}
   \;e^{-\frac{i}{\hbar}\hat{T}\,\Delta t}
   \;e^{-\frac{i}{\hbar}\hat{V}\,\Delta t/2}
   \; + \;\mathcal{O}(\Delta t^3).

Two important consequences:

- **Second-order global accuracy**: local error is :math:`\mathcal{O}(\Delta t^3)`, hence global error after :math:`N\sim T/\Delta t` steps is typically :math:`\mathcal{O}(\Delta t^2)`.
- **Unitarity (norm preservation)**: each factor is unitary (exponential of a Hermitian operator times :math:`-i`), so the method preserves :math:`\|\psi\|_2` up to numerical roundoff.

Why it is fast (Fourier diagonalizes the kinetic term)
------------------------------------------------------
In coordinate space, the potential exponential is pointwise multiplication:

.. math::

   \left(e^{-\frac{i}{\hbar}V(x)\Delta t/2}\psi\right)(x)
   = e^{-\frac{i}{\hbar}V(x)\Delta t/2}\,\psi(x).

The kinetic exponential is easiest in momentum (or wave-number) space. Let :math:`\tilde\psi(k)=\mathcal{F}[\psi](k)`.
Since :math:`\hat{p}=\hbar k` on plane waves, one has

.. math::

   \left(e^{-\frac{i}{\hbar}\hat{T}\Delta t}\tilde\psi\right)(k)
   = e^{-\frac{i}{\hbar}\frac{(\hbar k)^2}{2m}\Delta t}\,\tilde\psi(k)
   = e^{-i\frac{\hbar k^2}{2m}\Delta t}\,\tilde\psi(k).

So one time step is:

1. Multiply in :math:`x` space by :math:`e^{-\frac{i}{\hbar}V(x)\Delta t/2}`.
2. FFT to :math:`k` space.
3. Multiply in :math:`k` space by :math:`e^{-i\frac{\hbar k^2}{2m}\Delta t}`.
4. Inverse FFT back to :math:`x` space.
5. Multiply again by :math:`e^{-\frac{i}{\hbar}V(x)\Delta t/2}`.

This costs :math:`\mathcal{O}(N\log N)` per step (FFT) and is typically much faster than finite-difference implicit solvers for large grids.

Practical notes
---------------
- **Boundary conditions**: FFT implies periodic boundaries; use a large domain + absorbing layers (complex absorbing potential) if needed.
- **Time step**: choose :math:`\Delta t` small enough to resolve phase oscillations from both :math:`T` and :math:`V`.
- **Normalization**: the algorithm is norm-preserving, but a sanity check :math:`\int |\psi|^2 dx` helps catch discretization mistakes.

Animated example (Gaussian wavepacket in a harmonic trap)
---------------------------------------------------------
Below is an original animation generated from a 1D harmonic potential :math:`V(x)=\tfrac{1}{2}m\omega^2x^2`.
A displaced Gaussian wavepacket oscillates in the trap while spreading slightly due to dispersion.

.. figure:: ../_static_files/images/split_operator_wavepacket.gif
   :alt: Split-operator propagation of a Gaussian wavepacket in a harmonic potential
   :align: center
   :figwidth: 85%

   Probability density :math:`|\psi(x,t)|^2` evolved via the Strang split-operator method.

Step-by-step Python (1D)
------------------------
This is the minimal structure of a 1D implementation.

1. Choose a grid :math:`x_j` and its FFT wave-number grid :math:`k_j`.
2. Build :math:`V(x)` and precompute phase factors

   .. math::

      P_V(x)=e^{-\frac{i}{\hbar}V(x)\Delta t/2},\qquad
      P_T(k)=e^{-i\frac{\hbar k^2}{2m}\Delta t}.

3. Iterate the Strang step:

   .. math::

      \psi \leftarrow P_V\,\psi,\quad
      	ilde\psi\leftarrow \mathcal{F}[\psi],\quad
      	ilde\psi\leftarrow P_T\,\tilde\psi,\quad
      \psi\leftarrow \mathcal{F}^{-1}[\tilde\psi],\quad
      \psi \leftarrow P_V\,\psi.

Concrete snippet:

.. code-block:: python

   import numpy as np

   # Step 1: grid
   n = 1024
   x_max = 12.0
   x = np.linspace(-x_max, x_max, n, endpoint=False)
   dx = x[1] - x[0]
   k = 2*np.pi*np.fft.fftfreq(n, d=dx)

   # Parameters
   hbar = 1.0
   m = 1.0
   dt = 0.02

   # Step 2: potential and phases (example: harmonic)
   omega = 1.0
   V = 0.5*m*omega**2*x**2
   P_V = np.exp(-1j*V*(dt/2)/hbar)
   P_T = np.exp(-1j*(hbar*k**2)*dt/(2*m))

   # Step 3: initial state (example: Gaussian)
   psi = np.exp(-(x-4.0)**2/2.0) * np.exp(1j*0.0*x)
   psi = psi / np.sqrt(np.sum(np.abs(psi)**2)*dx)

   # Step 4: one Strang step
   psi = P_V * psi
   psi_k = np.fft.fft(psi)
   psi_k = P_T * psi_k
   psi = np.fft.ifft(psi_k)
   psi = P_V * psi

Reproducibility
---------------
The GIF was generated by the script:

.. literalinclude:: ../_static_files/codes/split_operator_wavepacket.py
   :language: python
   :linenos:

Run it to regenerate assets (it writes into ``docs/_static_files/images``).

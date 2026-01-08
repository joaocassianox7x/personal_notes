Double Pendulum via Lagrangian Mechanics
=======================================

Goal: derive the equations of motion with the Lagrangian, then simulate two nearly identical initial conditions to show chaotic divergence.

Geometry and coordinates
------------------------
- Two point masses :math:`m_1, m_2` on massless rods of lengths :math:`\ell_1, \ell_2`.
- Generalized coordinates: angles :math:`\theta_1, \theta_2` measured from the vertical.

.. image:: ../_static_files/images/double_pendulum_geometry.png
    :alt: Geometry of the double pendulum showing angles and link lengths
    :width: 80%
    :align: center

Positions (taking the pivot as the origin, downward is negative :math:`y`):

.. math::

    x_1 = \ell_1 \sin \theta_1, \quad y_1 = -\ell_1 \cos \theta_1 \\
    x_2 = x_1 + \ell_2 \sin \theta_2, \quad y_2 = y_1 - \ell_2 \cos \theta_2

Velocities:

.. math::

    \dot{x}_1 = \ell_1 \dot{\theta}_1 \cos \theta_1, \quad \dot{y}_1 = \ell_1 \dot{\theta}_1 \sin \theta_1 \\
    \dot{x}_2 = \dot{x}_1 + \ell_2 \dot{\theta}_2 \cos \theta_2, \quad \dot{y}_2 = \dot{y}_1 + \ell_2 \dot{\theta}_2 \sin \theta_2

Kinetic and potential energy
----------------------------
.. math::

    T = \tfrac{1}{2} m_1 (\dot{x}_1^2 + \dot{y}_1^2) + \tfrac{1}{2} m_2 (\dot{x}_2^2 + \dot{y}_2^2)

.. math::

    V = m_1 g y_1 + m_2 g y_2 = - m_1 g \ell_1 \cos\theta_1 - m_2 g (\ell_1 \cos\theta_1 + \ell_2 \cos\theta_2)

The Lagrangian is :math:`\mathcal{L} = T - V`.

Equations of motion (standard form)
------------------------------------
After simplifying Euler–Lagrange equations for :math:`\theta_1, \theta_2` and defining :math:`\omega_1 = \dot{\theta}_1`, :math:`\omega_2 = \dot{\theta}_2`:

.. math::

    \dot{\theta}_1 = \omega_1, \qquad \dot{\theta}_2 = \omega_2

.. math::
    \dot{\omega}_1 = \frac{-g (2 m_1 + m_2) \sin\theta_1 - m_2 g \sin(\theta_1 - 2\theta_2) - 2 \sin(\theta_1 - \theta_2) m_2 (\omega_2^2 \ell_2 + \omega_1^2 \ell_1 \cos(\theta_1 - \theta_2))}{\ell_1 (2 m_1 + m_2 - m_2 \cos(2\theta_1 - 2\theta_2))}

.. math::
    \dot{\omega}_2 = \frac{2 \sin(\theta_1 - \theta_2) (\omega_1^2 \ell_1 (m_1+m_2) + g (m_1 + m_2) \cos\theta_1 + \omega_2^2 \ell_2 m_2 \cos(\theta_1 - \theta_2))}{\ell_2 (2 m_1 + m_2 - m_2 \cos(2\theta_1 - 2\theta_2))}

These ODEs are stiff when the denominator approaches zero; a small time step and stable integrator (e.g., RK4) keeps the simulation stable for moderate times.

Numerical simulation recipe
---------------------------
1. Integrate :math:`[\theta_1, \theta_2, \omega_1, \omega_2]` with RK4.
2. Use two initial states that differ by a tiny perturbation to expose sensitivity.
3. Render both pendulums on the same plot and export to GIF.

Visualization results
---------------------
- Both runs start with :math:`\theta_1 = \pi/2`, :math:`\theta_2 = \pi/2 + 0.01`, and the second run perturbs :math:`\theta_2` by an extra :math:`10^{-3}`.
- Despite near-identical starts, trajectories quickly diverge.

.. image:: ../_static_files/images/double_pendulum_divergence.png
   :alt: Angle divergence between two nearly identical initial conditions
   :width: 100%
   :align: center

.. image:: ../_static_files/images/double_pendulum.gif
   :alt: Double pendulum animation showing two trajectories diverging
   :width: 100%
   :align: center

How to reproduce the figures
----------------------------
Run the script that integrates the ODEs and builds the GIF and PNG:

.. code-block:: bash

   python docs/_static_files/codes/double_pendulum.py

Key points from the script
--------------------------
- Uses RK4 with :math:`\Delta t = 0.005` over 20 seconds.
- Converts angles to Cartesian coordinates for plotting rods and masses.
- Uses ``matplotlib.animation.PillowWriter`` to emit ``double_pendulum.gif``.
- Also plots the angle differences :math:`|\theta_1^{(A)}-\theta_1^{(B)}|` and :math:`|\theta_2^{(A)}-\theta_2^{(B)}|` over time.

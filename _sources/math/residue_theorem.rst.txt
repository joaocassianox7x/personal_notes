Residue Theorem
===============

Summary (Arfken)
----------------
Given a function :math:`f(z)` that is analytic on and inside a closed contour :math:`C` except for isolated poles :math:`z_k`, the contour integral obeys the residue theorem (Arfken, Weber, Harris, *Mathematical Methods for Physicists*):

.. math::

   \oint_C f(z)\,dz = 2\pi i \sum_k \operatorname{Res}\big(f, z_k\big).

Residues capture the coefficient of :math:`(z - z_k)^{-1}` in the Laurent series of :math:`f` about :math:`z_k`.

.. figure:: ../_static_files/images/residue_contour.svg
   :alt: Contour integral around simple poles
   :align: center
   :figwidth: 60%

   Simple contour enclosing two simple poles; only enclosed residues contribute to the integral.

Residue formulas
----------------
- Simple pole at :math:`z_0`: :math:`\operatorname{Res}(f,z_0) = \lim_{z\to z_0} (z - z_0) f(z)`.
- Higher-order pole (order :math:`m`): :math:`\operatorname{Res}(f,z_0) = \tfrac{1}{(m-1)!} \lim_{z\to z_0} \tfrac{d^{m-1}}{dz^{m-1}} \big[(z - z_0)^m f(z)\big]`.
- At infinity (if useful for rational functions): :math:`\operatorname{Res}(f,\infty)` equals minus the coefficient of :math:`z^{-1}` in the Laurent series of :math:`f` at infinity.

Practical checklist
-------------------
1. **Choose contour:** match decay/symmetry of the integrand (unit circle for trigonometric, semicircle for decaying exponentials, keyholes for branch cuts).
2. **Rewrite integrand:** express in :math:`z` with :math:`z = e^{ix}` or other substitutions; convert differentials (e.g., :math:`dx = \tfrac{dz}{i z}`).
3. **Locate poles inside contour:** classify orders and check if any cancel.
4. **Compute residues:** use limit formulas for simple poles or quick partial fractions when possible.
5. **Sum residues and multiply by :math:`2\pi i`:** then take real/imag part if needed for the original real integral.

Notes on contour choices
------------------------
- **Unit circle**: trigonometric integrals via :math:`z=e^{ix}`.
- **Upper/lower semicircle**: Fourier-type integrals with exponential decay in half-planes.
- **Keyhole**: branch cuts on the positive real axis for logarithms or fractional powers.
- **Indented contours**: avoid poles on the real axis by small semicircular detours.

Methodology
-----------
1. Choose a contour that encloses the singularities relevant to the real integral (often the unit circle for trigonometric integrals using :math:`z = e^{ix}`).
2. Rewrite the integrand as a complex function :math:`f(z)` and express :math:`dx` in terms of :math:`dz`.
3. Identify poles inside the contour and compute their residues (by series expansion or the limit formula for simple poles).
4. Apply :math:`\oint_C f(z) dz = 2\pi i \sum \operatorname{Res}(f, z_k)` and take the real part if the original integral was real-valued.

Example: :math:`\int_0^{2\pi} \cos^{2n}(x)\,dx`
-------------------------------------------------
Let :math:`z = e^{ix}` so :math:`\cos x = \tfrac{1}{2}(z + z^{-1})` and :math:`dx = \tfrac{dz}{i z}` on the unit circle :math:`|z| = 1`.

.. math::

   I_n = \int_0^{2\pi} \cos^{2n}(x)\,dx = \Re\left( \oint_{|z|=1} \frac{1}{i z} \left( \frac{z + z^{-1}}{2} \right)^{2n} dz \right).

Expand the binomial :math:`(z + z^{-1})^{2n} = \sum_{k=0}^{2n} \binom{2n}{k} z^{2k-2n}`. Then

.. math::

   \frac{1}{i z} \left( \frac{z + z^{-1}}{2} \right)^{2n} = \frac{1}{i 2^{2n}} \sum_{k=0}^{2n} \binom{2n}{k} z^{2k-2n-1}.

The only term contributing a residue at :math:`z = 0` is the :math:`z^{-1}` term, which occurs when :math:`2k - 2n - 1 = -1`, i.e., :math:`k = n`. Hence

.. math::

   \operatorname{Res}\left( \frac{1}{i z} \left( \frac{z + z^{-1}}{2} \right)^{2n}, 0 \right) = \frac{1}{i 2^{2n}} \binom{2n}{n}.

Applying the residue theorem yields

.. math::

   I_n = 2\pi i \cdot \frac{1}{i 2^{2n}} \binom{2n}{n} = \frac{2\pi}{2^{2n}} \binom{2n}{n} = 2\pi \frac{(2n)!}{4^{n} (n!)^2}.

This matches the standard reduction formula and provides a quick contour-integral evaluation for even powers of cosine.

Additional examples to practice
-------------------------------
- :math:`\int_{-\infty}^{\infty} \frac{dx}{x^2 + a^2}` using a semicircle in the upper half-plane.
- :math:`\int_0^{\infty} \frac{\cos bx}{x^2 + a^2}\,dx` via Jordan's lemma.
- :math:`\int_0^{2\pi} \frac{d\theta}{5 - 4 \cos \theta}` via the unit circle and a simple pole.

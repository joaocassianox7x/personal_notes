Residue Theorem
===============

Summary (Arfken)
----------------
Given a function :math:`f(z)` that is analytic on and inside a closed contour :math:`C` except for isolated poles :math:`z_k`, the contour integral obeys the residue theorem (Arfken, Weber, Harris, *Mathematical Methods for Physicists*):

.. math::

   \oint_C f(z)\,dz = 2\pi i \sum_k \operatorname{Res}\big(f, z_k\big).

Residues capture the coefficient of :math:`(z - z_k)^{-1}` in the Laurent series of :math:`f` about :math:`z_k`.

Intuition (why it works)
------------------------
If :math:`f` is analytic in a punctured neighborhood of :math:`z_0`, it has a Laurent expansion

.. math::

   f(z) = \sum_{n=-\infty}^{\infty} a_n (z-z_0)^n.

When you integrate term-by-term around a small circle :math:`|z-z_0|=\rho`, all powers vanish except the :math:`n=-1` term:

.. math::

   \oint_{|z-z_0|=\rho} (z-z_0)^n\,dz =
   \begin{cases}
   2\pi i, & n=-1,\\
   0, & n\neq -1.
   \end{cases}

So the local contribution of an isolated singularity is exactly :math:`2\pi i\,a_{-1}`. Summing over all enclosed singularities gives the global contour integral.

.. figure:: ../_static_files/images/residue_contour.svg
   :alt: Contour integral around simple poles
   :align: center
   :figwidth: 60%

   Simple contour enclosing two simple poles; only enclosed residues contribute to the integral.

Residue formulas
----------------
- Simple pole at :math:`z_0`: :math:`\operatorname{Res}(f,z_0) = \lim_{z\to z_0} (z - z_0) f(z)`.
- Higher-order pole (order :math:`m`): :math:`\operatorname{Res}(f,z_0) = \tfrac{1}{(m-1)!} \lim_{z\to z_0} \tfrac{d^{m-1}}{dz^{m-1}} \big[(z - z_0)^m f(z)\big]`.
- At infinity (often useful for rational functions): :math:`\operatorname{Res}(f,\infty)` equals minus the coefficient of :math:`z^{-1}` in the Laurent series of :math:`f` at infinity.

Equivalent identities (handy in practice)
-----------------------------------------
For a meromorphic function on the Riemann sphere (e.g., a rational function), the residues satisfy

.. math::

    \sum_{k\in\mathbb{C}} \operatorname{Res}(f, z_k) + \operatorname{Res}(f,\infty) = 0.

For rational :math:`f(z)=\tfrac{p(z)}{q(z)}` with :math:`\deg q \ge \deg p + 2`, one has :math:`\operatorname{Res}(f,\infty)=0`, hence the sum of all finite residues is zero.

Fast residue methods (beyond the limit formula)
-----------------------------------------------
- **Series match (Laurent/Taylor)**: expand factors until the :math:`(z-z_0)^{-1}` term appears.
- **Partial fractions** (rational functions): isolate the :math:`\tfrac{A}{z-z_0}` term.
- **Simple pole at a simple root of the denominator**:

   If :math:`f(z)=\tfrac{g(z)}{h(z)}` and :math:`h(z_0)=0` with :math:`h'(z_0)\neq 0`, then

   .. math::

       \operatorname{Res}(f,z_0) = \frac{g(z_0)}{h'(z_0)}.

- **Logarithmic derivative trick** (common in products):

   If :math:`f(z)=\tfrac{g'(z)}{g(z)}` and :math:`g` has a simple zero at :math:`z_0`, then :math:`\operatorname{Res}(f,z_0)=1`.

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

Worked examples
---------------

Example 1: :math:`\int_{-\infty}^{\infty} \frac{dx}{x^2+a^2}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assume :math:`a>0`. Consider the semicircular contour in the upper half-plane: the real segment from :math:`-R` to :math:`R` plus the arc :math:`z=Re^{i\theta}`, :math:`\theta\in[0,\pi]`.

Let

.. math::

   f(z) = \frac{1}{z^2+a^2} = \frac{1}{(z-ia)(z+ia)}.

There is a single pole inside the contour at :math:`z=ia`. Its residue is

.. math::

   \operatorname{Res}(f, ia)
   = \lim_{z\to ia} (z-ia)\frac{1}{(z-ia)(z+ia)}
   = \frac{1}{2ia}.

Hence

.. math::

   \oint_C f(z)\,dz = 2\pi i\,\frac{1}{2ia} = \frac{\pi}{a}.

On the arc, :math:`|f(z)|\le \tfrac{1}{R^2-a^2}` so the arc contribution vanishes as :math:`R\to\infty`. Therefore

.. math::

   \int_{-\infty}^{\infty} \frac{dx}{x^2+a^2} = \frac{\pi}{a}.


Example 2: :math:`\int_0^{\infty} \frac{\cos(bx)}{x^2+a^2}\,dx`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assume :math:`a>0` and first take :math:`b>0`. Consider

.. math::

   I(b)=\int_{-\infty}^{\infty} \frac{e^{ibx}}{x^2+a^2}\,dx.

Close the contour in the upper half-plane. By Jordan's lemma the arc integral vanishes (because :math:`e^{ibz}=e^{ibx-b\,\Im z}` decays when :math:`\Im z>0`). The only enclosed pole is :math:`z=ia` with residue

.. math::

   \operatorname{Res}\left(\frac{e^{ibz}}{z^2+a^2}, ia\right)
   = \frac{e^{ib(ia)}}{2ia}
   = \frac{e^{-ab}}{2ia}.

Therefore

.. math::

   I(b)=2\pi i\,\frac{e^{-ab}}{2ia}=\frac{\pi}{a}e^{-ab}.

Taking the real part gives

.. math::

   \int_{-\infty}^{\infty} \frac{\cos(bx)}{x^2+a^2}\,dx = \frac{\pi}{a}e^{-ab}.

Since the integrand is even,

.. math::

   \int_{0}^{\infty} \frac{\cos(bx)}{x^2+a^2}\,dx = \frac{\pi}{2a}e^{-ab}\qquad (b>0).

For general real :math:`b`, symmetry yields :math:`e^{-ab}\to e^{-a|b|}`.


Example 3: :math:`\int_0^{2\pi} \frac{d\theta}{a+b\cos\theta}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assume :math:`a>|b|>0` so the denominator never vanishes on the real interval.

Set :math:`z=e^{i\theta}` so :math:`d\theta = \tfrac{dz}{iz}` and :math:`\cos\theta=\tfrac{1}{2}\left(z+z^{-1}\right)`. Then

.. math::

   \int_0^{2\pi} \frac{d\theta}{a+b\cos\theta}
   = \oint_{|z|=1} \frac{1}{a+\tfrac{b}{2}(z+z^{-1})}\,\frac{dz}{iz}
   = \oint_{|z|=1} \frac{2}{i\,(b z^2 + 2 a z + b)}\,dz.

The poles are the roots of :math:`b z^2 + 2 a z + b=0`:

.. math::

   z_{\pm} = \frac{-a \pm \sqrt{a^2-b^2}}{b}.

For :math:`a>|b|`, one root lies inside the unit circle and the other outside; the inside pole is

.. math::

   z_- = \frac{-a + \sqrt{a^2-b^2}}{b},\qquad |z_-|<1.

Since it is a simple pole, using :math:`\operatorname{Res}\big(\tfrac{1}{P(z)},z_-\big)=\tfrac{1}{P'(z_-)}` for :math:`P(z)=b z^2 +2az + b`,

.. math::

   \operatorname{Res}\left(\frac{2}{i\,P(z)}, z_-\right)
   = \frac{2}{i\,P'(z_-)}
   = \frac{2}{i\,(2b z_- + 2a)}.

But :math:`2b z_- + 2a = 2\sqrt{a^2-b^2}`. Therefore

.. math::

   \int_0^{2\pi} \frac{d\theta}{a+b\cos\theta}
   = 2\pi i\,\frac{2}{i\,2\sqrt{a^2-b^2}}
   = \frac{2\pi}{\sqrt{a^2-b^2}}.


Example 4: :math:`\int_0^{2\pi} \cos^{2n}(x)\,dx`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


Example 5 (keyhole/branch cut): :math:`\int_0^{\infty} \frac{x^{\alpha-1}}{1+x}\,dx`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This classical contour integral connects residues to the Beta/Gamma functions. Assume :math:`0<\alpha<1` and define the branch

.. math::

   z^{\alpha-1} = e^{(\alpha-1)\Log z},\qquad \Arg z\in(0,2\pi).

Use a keyhole contour encircling the positive real axis (with a small circle around the origin and a large circle at infinity). Consider

.. math::

   f(z)=\frac{z^{\alpha-1}}{1+z}.

There is a simple pole at :math:`z=-1` with residue

.. math::

   \operatorname{Res}(f,-1)=\lim_{z\to-1}(z+1)\frac{z^{\alpha-1}}{1+z}=(-1)^{\alpha-1}=e^{i\pi(\alpha-1)}.

The integral over the two sides of the cut differ by a phase. On the upper side :math:`\Arg z = 0`, so :math:`z^{\alpha-1}=x^{\alpha-1}`; on the lower side :math:`\Arg z = 2\pi`, so :math:`z^{\alpha-1}=e^{2\pi i(\alpha-1)}x^{\alpha-1}`. Taking the contour orientation into account, one obtains

.. math::

   \oint_C f(z)\,dz
   = \left(1-e^{2\pi i(\alpha-1)}\right)\int_0^{\infty}\frac{x^{\alpha-1}}{1+x}\,dx
   = \left(1-e^{2\pi i\alpha}\right)\int_0^{\infty}\frac{x^{\alpha-1}}{1+x}\,dx.

By residues, the same contour integral equals

.. math::

   \oint_C f(z)\,dz = 2\pi i\,\operatorname{Res}(f,-1)=2\pi i\,e^{i\pi(\alpha-1)}.

Using :math:`1-e^{2\pi i\alpha} = -2i e^{i\pi\alpha}\sin(\pi\alpha)`, it follows that

.. math::

   \int_0^{\infty}\frac{x^{\alpha-1}}{1+x}\,dx
   = \frac{\pi}{\sin(\pi\alpha)}\qquad (0<\alpha<1).


Example 6 (pole on the integration path): :math:`\mathrm{PV}\int_{-\infty}^{\infty} \frac{\sin x}{x}\,dx`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This illustrates *indented contours* and principal values. Consider

.. math::

   J = \mathrm{PV}\int_{-\infty}^{\infty} \frac{e^{ix}}{x}\,dx.

Close in the upper half-plane and detour around the pole at the origin with a small semicircle of radius :math:`\varepsilon` above the real axis. There are no poles in the upper half-plane, but the indentation contributes

.. math::

   \int_{\text{indent}} \frac{e^{iz}}{z}\,dz \xrightarrow[\varepsilon\to 0]{} i\pi.

Hence :math:`J=i\pi`. Taking the imaginary part gives the standard result

.. math::

   \mathrm{PV}\int_{-\infty}^{\infty} \frac{\sin x}{x}\,dx = \pi.

Additional examples to practice
-------------------------------
- Compute residues for higher-order poles, e.g. :math:`\operatorname{Res}\big(\tfrac{e^z}{z^3},0\big)` via series.
- Evaluate :math:`\int_0^{2\pi} \tfrac{d\theta}{5-4\cos\theta}` and check it matches :math:`\tfrac{2\pi}{\sqrt{5^2-4^2}}`.
- Use :math:`\operatorname{Res}(f,\infty)` to verify that the sum of all residues of a rational function is zero.

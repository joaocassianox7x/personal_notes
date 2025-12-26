Limits
======

Intuitive view
--------------
:math:`\lim_{x \to a} f(x) = L` means :math:`f(x)` can be made arbitrarily close to :math:`L` by taking :math:`x` sufficiently close to :math:`a` (not necessarily equal).

Epsilon–delta definition
------------------------
For all :math:`\varepsilon > 0`, there exists :math:`\delta > 0` such that if :math:`0 < |x - a| < \delta`, then :math:`|f(x) - L| < \varepsilon`.

Basic facts
-----------
- Linearity: :math:`\lim (a f + b g) = a \lim f + b \lim g` (when limits exist).
- Product: :math:`\lim (f g) = (\lim f)(\lim g)`.
- Quotient: :math:`\lim (f / g) = (\lim f) / (\lim g)` if :math:`\lim g \neq 0`.
- Squeeze: if :math:`g(x) \le f(x) \le h(x)` and :math:`\lim g = \lim h = L`, then :math:`\lim f = L`.

One-sided limits
----------------
- Right-hand: :math:`\lim_{x \to a^+} f(x)`.
- Left-hand: :math:`\lim_{x \to a^-} f(x)`.
The (two-sided) limit exists iff both one-sided limits exist and are equal.

Common limit tricks
-------------------
- Factor and cancel to resolve indeterminate forms.
- Multiply by a conjugate for expressions with roots.
- Use special limits: :math:`\lim_{x \to 0} \dfrac{\sin x}{x} = 1`, :math:`\lim_{x \to 0} (1 + x)^{1/x} = e`.

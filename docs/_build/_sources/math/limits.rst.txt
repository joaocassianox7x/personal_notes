Limits
======

Why limits matter
-----------------
The idea of limit is the foundation of calculus. It lets us describe what happens to
:math:`f(x)` when :math:`x` gets close to a point, even if :math:`f(a)` is not defined
or is not the value that controls the nearby behavior.

In practice, limits are used to define:

- continuity,
- derivatives,
- integrals,
- asymptotic behavior.

Informal idea
-------------
We write

.. math::

   \lim_{x \to a} f(x) = L

when :math:`f(x)` gets arbitrarily close to :math:`L` as :math:`x` gets arbitrarily
close to :math:`a`, with :math:`x \neq a`.

Important: the limit depends on the values of :math:`f(x)` near :math:`a`, not
necessarily on the value of :math:`f(a)` itself.

Example:

.. math::

   f(x)=\frac{x^2-1}{x-1}, \quad x \neq 1.

Since :math:`x^2-1=(x-1)(x+1)`, we have :math:`f(x)=x+1` for :math:`x \neq 1`, so

.. math::

   \lim_{x \to 1} \frac{x^2-1}{x-1} = 2,

although the original formula is not defined at :math:`x=1`.

Formal definition (epsilon-delta)
---------------------------------
The statement

.. math::

   \lim_{x \to a} f(x) = L

means that for every :math:`\varepsilon > 0`, there exists :math:`\delta > 0` such
that

.. math::

   0 < |x-a| < \delta \implies |f(x)-L| < \varepsilon.

Interpretation:

- :math:`\varepsilon` controls how close :math:`f(x)` must be to :math:`L`.
- :math:`\delta` controls how close :math:`x` must be to :math:`a`.
- The condition :math:`0 < |x-a|` excludes the point :math:`x=a` itself.

One-sided limits
----------------
Sometimes the left and right behaviors must be analyzed separately.

- Left-hand limit: :math:`\lim_{x \to a^-} f(x)`.
- Right-hand limit: :math:`\lim_{x \to a^+} f(x)`.

The two-sided limit exists if and only if both one-sided limits exist and are equal:

.. math::

   \lim_{x \to a} f(x)=L
   \iff
   \lim_{x \to a^-} f(x)=\lim_{x \to a^+} f(x)=L.

Example:

.. math::

   f(x)=\frac{|x|}{x}.

Then

.. math::

   \lim_{x \to 0^-} \frac{|x|}{x}=-1, \qquad
   \lim_{x \to 0^+} \frac{|x|}{x}=1.

Since the one-sided limits are different, :math:`\lim_{x \to 0} \frac{|x|}{x}` does not exist.

Basic properties of limits
--------------------------
Assume :math:`\lim_{x \to a} f(x)=L` and :math:`\lim_{x \to a} g(x)=M`.
Then:

- Constant: :math:`\lim_{x \to a} c = c`.
- Sum: :math:`\lim_{x \to a} (f+g)=L+M`.
- Difference: :math:`\lim_{x \to a} (f-g)=L-M`.
- Scalar multiple: :math:`\lim_{x \to a} (cf)=cL`.
- Product: :math:`\lim_{x \to a} (fg)=LM`.
- Quotient: :math:`\lim_{x \to a} \dfrac{f}{g}=\dfrac{L}{M}` if :math:`M \neq 0`.
- Power: :math:`\lim_{x \to a} [f(x)]^n = L^n` for :math:`n \in \mathbb{N}`.
- Root: :math:`\lim_{x \to a} \sqrt[n]{f(x)} = \sqrt[n]{L}` when the expression makes sense.

Useful consequence:

- Polynomials are evaluated by direct substitution.
- Rational functions are evaluated by direct substitution whenever the denominator
  does not vanish.

Order theorem and squeeze theorem
---------------------------------
If :math:`f(x) \le g(x)` for all :math:`x` near :math:`a`, and both limits exist, then

.. math::

   \lim_{x \to a} f(x) \le \lim_{x \to a} g(x).

A very useful consequence is the squeeze theorem.

If

.. math::

   g(x) \le f(x) \le h(x)

for :math:`x` near :math:`a`, and

.. math::

   \lim_{x \to a} g(x)=\lim_{x \to a} h(x)=L,

then

.. math::

   \lim_{x \to a} f(x)=L.

Classic example:

.. math::

   -|x| \le x\sin\left(\frac{1}{x}\right) \le |x|.

Since :math:`\lim_{x \to 0} (-|x|)=0` and :math:`\lim_{x \to 0} |x|=0`, we get

.. math::

   \lim_{x \to 0} x\sin\left(\frac{1}{x}\right)=0.

Limits and continuity
---------------------
A function :math:`f` is continuous at :math:`x=a` when:

- :math:`f(a)` exists,
- :math:`\lim_{x \to a} f(x)` exists,
- :math:`\lim_{x \to a} f(x)=f(a)`.

So continuity is exactly the statement that the nearby behavior and the function
value match.

Examples of continuous functions on their domains:

- polynomials,
- rational functions where the denominator is nonzero,
- trigonometric functions,
- exponential and logarithmic functions,
- roots where they are defined.

This gives the practical rule:
if :math:`f` is continuous at :math:`a`, then

.. math::

   \lim_{x \to a} f(x)=f(a).

Infinite limits
---------------
If :math:`f(x)` grows without bound as :math:`x` approaches :math:`a`, we write

.. math::

   \lim_{x \to a} f(x)=+\infty

or

.. math::

   \lim_{x \to a} f(x)=-\infty.

This does not mean the limit is a real number. It means the function becomes
arbitrarily large in magnitude.

Example:

.. math::

   \lim_{x \to 2} \frac{1}{(x-2)^2}=+\infty.

This indicates a vertical asymptote at :math:`x=2`.

Limits at infinity
------------------
We also study the behavior of :math:`f(x)` when :math:`x` becomes very large in
absolute value:

.. math::

   \lim_{x \to +\infty} f(x), \qquad \lim_{x \to -\infty} f(x).

If

.. math::

   \lim_{x \to \pm\infty} f(x)=L,

then the line :math:`y=L` is a horizontal asymptote.

For rational functions, compare the highest powers:

- degree(numerator) < degree(denominator): limit is :math:`0`.
- degree(numerator) = degree(denominator): limit is the ratio of leading coefficients.
- degree(numerator) > degree(denominator): no finite horizontal asymptote.

Example:

.. math::

   \lim_{x \to \infty} \frac{3x^2-1}{x^2+5x+2}=3.

Remarkable limits
-----------------
Some limits appear constantly and should be memorized.

1. Trigonometric fundamental limit:

   .. math::

      \lim_{x \to 0} \frac{\sin x}{x}=1.

2. Exponential limit:

   .. math::

      \lim_{x \to 0} (1+x)^{1/x}=e.

From these, several others follow:

.. math::

   \lim_{x \to 0} \frac{\tan x}{x}=1,
   \qquad
   \lim_{x \to 0} \frac{1-\cos x}{x}=0,
   \qquad
   \lim_{x \to 0} \frac{1-\cos x}{x^2}=\frac{1}{2},

and, by substitution,

.. math::

   \lim_{x \to 0} \frac{\sin(ax)}{ax}=1,
   \qquad
   \lim_{x \to 0} \frac{\sin(ax)}{x}=a.

Indeterminate forms
-------------------
Direct substitution sometimes gives expressions that do not determine the answer.
The most common indeterminate forms are:

- :math:`0/0`,
- :math:`\infty/\infty`,
- :math:`0 \cdot \infty`,
- :math:`\infty - \infty`,
- :math:`1^\infty`,
- :math:`0^0`,
- :math:`\infty^0`.

An indeterminate form is not the answer. It is a warning that more work is needed.

Most common techniques
----------------------
When direct substitution fails, the standard tools are:

1. Factor and cancel.
2. Multiply by the conjugate.
3. Put terms over a common denominator.
4. Divide by the highest power when :math:`x \to \infty`.
5. Use trigonometric identities and remarkable limits.
6. Apply the squeeze theorem.
7. Rewrite the expression before taking the limit.

Worked examples
---------------
Example 1: direct substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to 2} (x^2+3x-1).

Since polynomials are continuous,

.. math::

   \lim_{x \to 2} (x^2+3x-1)=2^2+3\cdot 2-1=9.

Example 2: factor and cancel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to 1} \frac{x^2-1}{x-1}.

Step 1. Factor the numerator:

.. math::

   x^2-1=(x-1)(x+1).

Step 2. Cancel the common factor for :math:`x \neq 1`:

.. math::

   \frac{x^2-1}{x-1}=x+1.

Step 3. Take the limit:

.. math::

   \lim_{x \to 1} \frac{x^2-1}{x-1}=\lim_{x \to 1}(x+1)=2.

Example 3: conjugate
~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to 4} \frac{\sqrt{x}-2}{x-4}.

Step 1. Multiply numerator and denominator by the conjugate:

.. math::

   \frac{\sqrt{x}-2}{x-4}\cdot\frac{\sqrt{x}+2}{\sqrt{x}+2}
   =\frac{x-4}{(x-4)(\sqrt{x}+2)}.

Step 2. Cancel :math:`x-4`:

.. math::

   \frac{\sqrt{x}-2}{x-4}=\frac{1}{\sqrt{x}+2}.

Step 3. Take the limit:

.. math::

   \lim_{x \to 4} \frac{\sqrt{x}-2}{x-4}=\frac{1}{4}.

Example 4: one-sided limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Study

.. math::

   \lim_{x \to 0} \frac{|x|}{x}.

For :math:`x>0`, :math:`|x|=x`, so :math:`\frac{|x|}{x}=1`.
For :math:`x<0`, :math:`|x|=-x`, so :math:`\frac{|x|}{x}=-1`.
Therefore,

.. math::

   \lim_{x \to 0^-} \frac{|x|}{x}=-1, \qquad
   \lim_{x \to 0^+} \frac{|x|}{x}=1.

The two-sided limit does not exist.

Example 5: squeeze theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to 0} x\sin\left(\frac{1}{x}\right).

Since :math:`-1 \le \sin(1/x) \le 1`, multiplying by :math:`|x|` gives

.. math::

   -|x| \le x\sin\left(\frac{1}{x}\right) \le |x|.

Now :math:`|x| \to 0`, so by the squeeze theorem,

.. math::

   \lim_{x \to 0} x\sin\left(\frac{1}{x}\right)=0.

Example 6: remarkable trigonometric limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to 0} \frac{\sin(3x)}{x}.

Rewrite as

.. math::

   \frac{\sin(3x)}{x}=3\cdot\frac{\sin(3x)}{3x}.

Since :math:`\lim_{u \to 0} \frac{\sin u}{u}=1`, we obtain

.. math::

   \lim_{x \to 0} \frac{\sin(3x)}{x}=3.

Example 7: limit at infinity of a rational function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to \infty} \frac{2x^2-3x+1}{x^2+5}.

Divide numerator and denominator by :math:`x^2`:

.. math::

   \frac{2x^2-3x+1}{x^2+5}=
   \frac{2-3/x+1/x^2}{1+5/x^2}.

As :math:`x \to \infty`, the terms :math:`1/x` and :math:`1/x^2` go to :math:`0`, so

.. math::

   \lim_{x \to \infty} \frac{2x^2-3x+1}{x^2+5}=2.

Example 8: infinite limit
~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \lim_{x \to 2} \frac{1}{(x-2)^2}.

Because :math:`(x-2)^2 > 0` and becomes arbitrarily small as :math:`x \to 2`, its
reciprocal becomes arbitrarily large. Hence

.. math::

   \lim_{x \to 2} \frac{1}{(x-2)^2}=+\infty.

Summary
-------
A solid limit calculation usually follows this order:

1. Try direct substitution.
2. If it is continuous, evaluate immediately.
3. If an indeterminate form appears, rewrite the expression.
4. Check whether one-sided limits are needed.
5. For :math:`x \to \infty`, compare dominant terms.
6. Use remarkable limits and squeeze when appropriate.

This is the standard toolbox used before derivatives and throughout the rest of calculus.

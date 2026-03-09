Integrals
=========

What is an integral?
--------------------
An integral is the reverse operation of differentiation.

- **Indefinite integral**: finds a family of antiderivatives.

  .. math::

     \int f(x)\,dx = F(x) + C,\quad \text{with }F'(x)=f(x).

- **Definite integral**: gives accumulated change (signed area under the curve).

  .. math::

     \int_a^b f(x)\,dx = F(b)-F(a).


Most common methods
-------------------
1. Integration by parts (IDV-VDI).
2. Substitution.
3. Trigonometric substitution.
4. Polynomial coefficients (partial fractions).


Integration by parts (IDV-VDI)
------------------------------
Formula:

.. math::

   \int u\,dv = uv - \int v\,du.

Use IDV-VDI as a mnemonic: choose what to **differentiate** as :math:`u` and what to **integrate** as :math:`dv`.

Worked examples (3)
~~~~~~~~~~~~~~~~~~~

Example 1: :math:`\int x e^x\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Choose :math:`u=x` and :math:`dv=e^x dx`.
2. Then :math:`du=dx` and :math:`v=e^x`.
3. Apply the formula:

   .. math::

      \int x e^x\,dx = x e^x - \int e^x\,dx = x e^x - e^x + C = e^x(x-1)+C.

Example 2: :math:`\int x\ln x\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Choose :math:`u=\ln x` and :math:`dv=x\,dx`.
2. Then :math:`du=\frac{1}{x}dx` and :math:`v=\frac{x^2}{2}`.
3. Apply the formula:

   .. math::

      \int x\ln x\,dx
      = \frac{x^2}{2}\ln x - \int \frac{x^2}{2}\cdot\frac{1}{x}\,dx
      = \frac{x^2}{2}\ln x - \frac{1}{2}\int x\,dx
      = \frac{x^2}{2}\ln x - \frac{x^2}{4} + C.

Example 3: :math:`\int e^x\sin x\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Let :math:`I=\int e^x\sin x\,dx`. Pick :math:`u=\sin x`, :math:`dv=e^x dx`.

   .. math::

      I = e^x\sin x - \int e^x\cos x\,dx.

2. Let :math:`J=\int e^x\cos x\,dx`. Again by parts with :math:`u=\cos x`, :math:`dv=e^x dx`:

   .. math::

      J = e^x\cos x + \int e^x\sin x\,dx = e^x\cos x + I.

3. Substitute :math:`J` into :math:`I` and solve:

   .. math::

      I = e^x\sin x - (e^x\cos x + I)
      \Rightarrow 2I = e^x(\sin x - \cos x)
      \Rightarrow I = \frac{e^x}{2}(\sin x - \cos x) + C.


Substitution
------------
Formula:

.. math::

   \text{If }u=g(x),\ du=g'(x)\,dx,\ \text{then }\int f(g(x))g'(x)\,dx=\int f(u)\,du.

Worked examples (3)
~~~~~~~~~~~~~~~~~~~

Example 1: :math:`\int 2x\cos(x^2)\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Set :math:`u=x^2`.
2. Then :math:`du=2x\,dx`.
3. Substitute:

   .. math::

      \int 2x\cos(x^2)\,dx = \int \cos u\,du = \sin u + C = \sin(x^2)+C.

Example 2: :math:`\int \frac{x}{x^2+1}\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Set :math:`u=x^2+1`.
2. Then :math:`du=2x\,dx`, so :math:`x\,dx=\frac{1}{2}du`.
3. Substitute:

   .. math::

      \int \frac{x}{x^2+1}\,dx
      = \frac{1}{2}\int \frac{1}{u}\,du
      = \frac{1}{2}\ln|u| + C
      = \frac{1}{2}\ln(x^2+1)+C.

Example 3: :math:`\int (3x+1)^5\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Set :math:`u=3x+1`.
2. Then :math:`du=3dx`, so :math:`dx=\frac{1}{3}du`.
3. Substitute:

   .. math::

      \int (3x+1)^5\,dx
      = \frac{1}{3}\int u^5\,du
      = \frac{1}{3}\cdot\frac{u^6}{6}+C
      = \frac{(3x+1)^6}{18}+C.


Trigonometric substitution
--------------------------
Typical substitutions:

- :math:`\sqrt{a^2-x^2}`: use :math:`x=a\sin\theta`.
- :math:`a^2+x^2`: use :math:`x=a\tan\theta`.
- :math:`\sqrt{x^2-a^2}`: use :math:`x=a\sec\theta`.

Worked examples (3)
~~~~~~~~~~~~~~~~~~~

Example 1: :math:`\int \frac{dx}{\sqrt{9-x^2}}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Use :math:`x=3\sin\theta`, so :math:`dx=3\cos\theta\,d\theta`.
2. Then :math:`\sqrt{9-x^2}=\sqrt{9-9\sin^2\theta}=3\cos\theta`.
3. Substitute:

   .. math::

      \int \frac{dx}{\sqrt{9-x^2}}
      = \int \frac{3\cos\theta\,d\theta}{3\cos\theta}
      = \int d\theta
      = \theta + C
      = \arcsin\!\left(\frac{x}{3}\right)+C.

Example 2: :math:`\int \frac{dx}{x^2+4}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Use :math:`x=2\tan\theta`, so :math:`dx=2\sec^2\theta\,d\theta`.
2. Then :math:`x^2+4=4\tan^2\theta+4=4\sec^2\theta`.
3. Substitute:

   .. math::

      \int \frac{dx}{x^2+4}
      = \int \frac{2\sec^2\theta\,d\theta}{4\sec^2\theta}
      = \frac{1}{2}\int d\theta
      = \frac{\theta}{2}+C
      = \frac{1}{2}\arctan\!\left(\frac{x}{2}\right)+C.

Example 3: :math:`\int \frac{dx}{\sqrt{x^2-9}}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Use :math:`x=3\sec\theta`, so :math:`dx=3\sec\theta\tan\theta\,d\theta`.
2. Then :math:`\sqrt{x^2-9}=\sqrt{9\sec^2\theta-9}=3\tan\theta`.
3. Substitute:

   .. math::

      \int \frac{dx}{\sqrt{x^2-9}}
      = \int \sec\theta\,d\theta
      = \ln|\sec\theta+\tan\theta| + C
      = \ln\!\left|x+\sqrt{x^2-9}\right| + C.


Polynomial coefficients (partial fractions)
--------------------------------------------
When the integrand is a rational function :math:`\frac{P(x)}{Q(x)}`, factor :math:`Q(x)` and write:

.. math::

   \frac{P(x)}{Q(x)} =
   \frac{A}{x-a} + \frac{B}{(x-a)^2} + \frac{Cx+D}{x^2+px+q} + \cdots

Then solve the unknown coefficients (:math:`A, B, C, D, \dots`) by matching polynomial coefficients.

Worked examples (3)
~~~~~~~~~~~~~~~~~~~

Example 1: :math:`\int \frac{3x+5}{x^2+3x+2}\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Factor denominator: :math:`x^2+3x+2=(x+1)(x+2)`.
2. Decompose:

   .. math::

      \frac{3x+5}{(x+1)(x+2)}=\frac{A}{x+1}+\frac{B}{x+2}.

   Multiply through:

   .. math::

      3x+5=A(x+2)+B(x+1)=(A+B)x+(2A+B).

   Match coefficients:

   .. math::

      A+B=3,\quad 2A+B=5 \Rightarrow A=2,\ B=1.

3. Integrate:

   .. math::

      \int \frac{3x+5}{x^2+3x+2}\,dx
      = \int \left(\frac{2}{x+1}+\frac{1}{x+2}\right)dx
      = 2\ln|x+1|+\ln|x+2|+C.

Example 2: :math:`\int \frac{5x+3}{x(x+1)^2}\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Decompose using repeated factor:

   .. math::

      \frac{5x+3}{x(x+1)^2}
      = \frac{A}{x}+\frac{B}{x+1}+\frac{C}{(x+1)^2}.

2. Multiply through and match coefficients:

   .. math::

      5x+3 = A(x+1)^2 + Bx(x+1) + Cx
      = (A+B)x^2 + (2A+B+C)x + A.

   So:

   .. math::

      A+B=0,\quad 2A+B+C=5,\quad A=3
      \Rightarrow B=-3,\ C=2.

3. Integrate:

   .. math::

      \int \frac{5x+3}{x(x+1)^2}\,dx
      = \int \left(\frac{3}{x}-\frac{3}{x+1}+\frac{2}{(x+1)^2}\right)dx
      = 3\ln|x|-3\ln|x+1|-\frac{2}{x+1}+C.

Example 3: :math:`\int \frac{2x^2+3x+4}{x(x^2+1)}\,dx`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Decompose with an irreducible quadratic:

   .. math::

      \frac{2x^2+3x+4}{x(x^2+1)} = \frac{A}{x}+\frac{Bx+C}{x^2+1}.

2. Multiply through and match coefficients:

   .. math::

      2x^2+3x+4 = A(x^2+1) + (Bx+C)x
      = (A+B)x^2 + Cx + A.

   So:

   .. math::

      A+B=2,\quad C=3,\quad A=4
      \Rightarrow B=-2,\ C=3.

3. Integrate:

   .. math::

      \int \frac{2x^2+3x+4}{x(x^2+1)}\,dx
      = \int \left(\frac{4}{x} + \frac{-2x+3}{x^2+1}\right)dx
      = 4\ln|x| - \ln(x^2+1) + 3\arctan x + C.

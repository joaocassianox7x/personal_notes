Stokes' Theorem
===============

What the theorem says
---------------------
Let :math:`S` be an oriented smooth surface with positively oriented boundary
:math:`C = \partial S`. For a vector field :math:`\mathbf{F}`, Stokes' theorem says

.. math::

   \oint_C \mathbf{F} \cdot d\mathbf{r}
   =
   \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n}\, dS.

The theorem converts:

- a closed line integral into a surface integral of the curl, or
- a surface integral of the curl into a line integral along the boundary.

It is one of the main tools for solving vector-calculus integrals.

Geometric meaning
-----------------
The left-hand side measures circulation of :math:`\mathbf{F}` around the boundary
:math:`C`.

The right-hand side measures how much the curl of :math:`\mathbf{F}` passes through
the surface :math:`S`.

So Stokes' theorem says:
circulation around the edge equals total rotation through the surface.

Orientation rule
----------------
The orientation of the boundary and the orientation of the surface must match.

Use the right-hand rule:

- point the thumb in the direction of the chosen normal :math:`\mathbf{n}`,
- the fingers curl in the positive direction of the boundary.

If the orientation is reversed, the sign of the integral changes.

When to use it
--------------
Stokes' theorem is especially useful when:

- the line integral is over a closed curve in 3D,
- the boundary comes from a curved surface but can be replaced by an easier surface
  with the same boundary,
- the curl is simpler than a direct parametrization of the curve,
- the surface integral of :math:`\nabla \times \mathbf{F}` is harder than the
  boundary integral.

Practical checklist
-------------------
1. Identify the vector field :math:`\mathbf{F}`.
2. Identify the curve :math:`C` and its orientation.
3. Compute :math:`\nabla \times \mathbf{F}`.
4. Choose the easier side of the theorem.
5. If needed, replace the surface by any simpler surface with the same boundary.
6. Check the orientation before integrating.

Worked examples
---------------
Example 1: closed line integral on a circle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \oint_C (-y, x, 0) \cdot d\mathbf{r},

where :math:`C` is the unit circle :math:`x^2 + y^2 = 1` in the plane :math:`z=0`,
oriented counterclockwise when viewed from above.

Step 1. Choose the vector field:

.. math::

   \mathbf{F}(x,y,z)=(-y,x,0).

Step 2. Compute the curl:

.. math::

   \nabla \times \mathbf{F}
   =
   \begin{pmatrix}
   \partial_y 0 - \partial_z x \\
   \partial_z(-y) - \partial_x 0 \\
   \partial_x x - \partial_y(-y)
   \end{pmatrix}
   =
   (0,0,2).

Step 3. Choose the easiest surface with boundary :math:`C`: the unit disk
:math:`D` in the plane :math:`z=0`.

Its upward unit normal is :math:`\mathbf{n}=(0,0,1)`.

Step 4. Apply Stokes' theorem:

.. math::

   \oint_C \mathbf{F}\cdot d\mathbf{r}
   =
   \iint_D (\nabla \times \mathbf{F})\cdot \mathbf{n}\, dS
   =
   \iint_D 2\, dA.

Step 5. Integrate over the disk:

.. math::

   \iint_D 2\, dA = 2 \cdot \text{Area}(D) = 2\pi.

Therefore,

.. math::

   \oint_C (-y, x, 0)\cdot d\mathbf{r} = 2\pi.


Example 2: triangle in an oblique plane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \oint_C (z, x, y)\cdot d\mathbf{r},

where :math:`C` is the boundary of the triangle with vertices
:math:`A=(1,0,0)`, :math:`B=(0,1,0)`, :math:`C=(0,0,1)`,
oriented in the order :math:`A \to B \to C \to A`.

Step 1. Choose the vector field:

.. math::

   \mathbf{F}(x,y,z)=(z,x,y).

Step 2. Compute the curl:

.. math::

   \nabla \times \mathbf{F}
   =
   \begin{pmatrix}
   \partial_y y - \partial_z x \\
   \partial_z z - \partial_x y \\
   \partial_x x - \partial_y z
   \end{pmatrix}
   =
   (1,1,1).

Step 3. Use the triangle itself as the surface.

An oriented area vector for the triangle is

.. math::

   \frac{1}{2}\big[(B-A)\times(C-A)\big].

Now

.. math::

   B-A = (-1,1,0), \qquad C-A = (-1,0,1),

so

.. math::

   (B-A)\times(C-A) = (1,1,1).

Hence the oriented area vector is

.. math::

   \frac{1}{2}(1,1,1).

Step 4. Since the curl is constant, the surface integral is curl dot oriented area:

.. math::

   \oint_C \mathbf{F}\cdot d\mathbf{r}
   =
   \iint_S (\nabla\times\mathbf{F})\cdot\mathbf{n}\, dS
   =
   (1,1,1)\cdot \frac{1}{2}(1,1,1).

Step 5. Compute:

.. math::

   (1,1,1)\cdot \frac{1}{2}(1,1,1)
   =
   \frac{1}{2}(1+1+1)
   =
   \frac{3}{2}.

Therefore,

.. math::

   \oint_C (z, x, y)\cdot d\mathbf{r} = \frac{3}{2}.


Example 3: use Stokes in the reverse direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute

.. math::

   \iint_S (\nabla \times \mathbf{F})\cdot \mathbf{n}\, dS,

where :math:`\mathbf{F}(x,y,z)=(-y,x,z)`, and :math:`S` is the upper hemisphere
:math:`x^2+y^2+z^2=9`, oriented with outward normal.

Step 1. Identify the boundary of the hemisphere.

The boundary curve is the circle

.. math::

   x^2+y^2=9, \qquad z=0,

oriented counterclockwise when viewed from above.

Step 2. Apply Stokes' theorem:

.. math::

   \iint_S (\nabla \times \mathbf{F})\cdot \mathbf{n}\, dS
   =
   \oint_{\partial S} \mathbf{F}\cdot d\mathbf{r}.

Step 3. Parametrize the boundary circle:

.. math::

   \mathbf{r}(t)=(3\cos t, 3\sin t, 0), \qquad 0 \le t \le 2\pi.

Then

.. math::

   \mathbf{r}'(t)=(-3\sin t, 3\cos t, 0).

Step 4. Evaluate the vector field on the curve:

.. math::

   \mathbf{F}(\mathbf{r}(t))=(-3\sin t, 3\cos t, 0).

Step 5. Compute the dot product:

.. math::

   \mathbf{F}(\mathbf{r}(t))\cdot \mathbf{r}'(t)
   =
   (-3\sin t)(-3\sin t) + (3\cos t)(3\cos t)
   =
   9.

Step 6. Integrate:

.. math::

   \oint_{\partial S} \mathbf{F}\cdot d\mathbf{r}
   =
   \int_0^{2\pi} 9\, dt
   =
   18\pi.

Therefore,

.. math::

   \iint_S (\nabla \times \mathbf{F})\cdot \mathbf{n}\, dS = 18\pi.


Summary
-------
Stokes' theorem is a method-selection tool:

- if the boundary curve is simple, use the line integral;
- if the curl and surface are simple, use the surface integral;
- if the original surface is complicated, replace it with an easier one having the
  same boundary and orientation.

That is why the theorem is so effective for solving integrals in vector calculus.

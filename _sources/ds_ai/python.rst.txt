Python for Data Science
=======================

Overview
--------
Python is a concise, readable language with a large scientific ecosystem. This
section mirrors the book's progression: core syntax, data structures, control
flow, functions, file I/O, then numerical methods and the scientific stack.

Development environment
-----------------------
The original material uses Spyder because it bundles the main scientific
packages and is lightweight. Spyder is still a good choice, but Jupyter and
VS Code are common alternatives. Prefer Python 3.10+ (3.11+ is faster), and
keep dependencies isolated with `venv` or `conda`.

Variables, types, and scope
---------------------------
Python is dynamically typed: the value determines the type.

.. code-block:: python

   a = 1        # int
   b = 3.1415   # float
   c = "python" # str
   d = True     # bool
   print(type(a), type(b), type(c), type(d))

Scope matters: variables defined inside a block are local to that block.

.. code-block:: python

   flag = True
   if flag:
       local_value = "ok"
   # local_value is undefined if flag is False

Basic packages and imports
--------------------------
The book introduces the standard library modules `random` and `time`.

.. code-block:: python

   import random
   import time

   x = random.random()  # float in [0, 1)
   t = time.time()      # epoch time in seconds

Conditional logic and boolean expressions
-----------------------------------------
Use `if`, `elif`, and `else` with boolean expressions. Combine conditions with
`and`, `or`, and `not`.

.. code-block:: python

   age = 26
   if age < 18:
       print("under 18")
   elif age == 18:
       print("exactly 18")
   else:
       print("over 18")

Sequences: lists, tuples, and matrices
--------------------------------------
Lists are mutable, tuples are immutable, and both are ordered sequences with
0-based indexing.

.. code-block:: python

   values = [1, 9, 4, 25]
   letters = ["a", "b", "c"]
   values.append(42)
   values.sort()
   print(len(values), max(values), min(values), sum(values))
   print(values[0], values[-1], values[0:2], values[::-1])

   coords = (1, 2, 3)  # tuple

Matrices are often represented as lists of lists (later, NumPy arrays).

.. code-block:: python

   matrix = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
   print(matrix[0][2])  # 3

Loops
-----
`while` repeats while a condition is true; `for` iterates over iterables.

.. code-block:: python

   num = 1
   while num < 5:
       print(num)
       num += 1

   for letter in "data":
       print(letter)

   for i in range(1, 4):
       print(i)

Nested `for` loops are a direct way to traverse matrices.

Functions
---------
Define functions with `def`, optionally with default parameters.

.. code-block:: python

   def square(x):
       return x ** 2

   def power(x, p=2):
       return x ** p

   print(square(7), power(3), power(2, 3))

Functions can be combined with random sampling, as in simple simulations.

File I/O
--------
Use `open()` with modes `r`, `w`, and `a`, or prefer a context manager to
auto-close files.

.. code-block:: python

   with open("texto.txt", "w", encoding="utf-8") as f:
       f.write("line 1\n")
       f.write("line 2\n")

   with open("texto.txt", "r", encoding="utf-8") as f:
       print(f.read())

Relative paths write to the current directory; absolute paths specify the full
location. For portable paths, use `pathlib.Path`.

Lambda and comprehensions
-------------------------
Lambda functions are small anonymous functions; comprehensions build lists (or
sets/dicts) concisely.

.. code-block:: python

   calc = lambda x: 3 * x + 1
   print(calc(4))

   nums = [1, 2, 3, 4, 5, 6]
   evens = [n for n in nums if n % 2 == 0]
   squares = [n ** 2 for n in nums]

Numerical methods in pure Python
--------------------------------
The book implements classic numerical methods without external libraries:

- Trapezoidal rule for integrals, useful for 1D problems.
- Monte Carlo integration, especially effective in higher dimensions.
- ODE solvers: Euler (second-order form), Runge-Kutta 4, and Verlet.
- Discrete Fourier transform (DFT) for frequency analysis.

These serve as a conceptual baseline before switching to optimized libraries.

Scientific Python stack
-----------------------
Matplotlib
~~~~~~~~~~
`matplotlib.pyplot` is used for plotting lines, scatter plots, subplots, and
saving figures. Common calls include `plot`, `legend`, `grid`, `subplot`, and
`savefig`. Labels can include LaTeX with raw strings, for example `r"$e^x$"`.

NumPy
~~~~~
NumPy arrays are faster and more memory-efficient than Python lists for numeric
data. Core tools include `np.arange`, `np.linspace`, `np.zeros`, vectorized
math, `np.random`, `np.fft`, and `numpy.linalg` (solve, eig, inv, det, dot,
inner, outer). Prefer vectorized operations over Python loops when possible.

SciPy
~~~~~
SciPy builds on NumPy and provides optimized routines for ODEs and integration.
The recommended ODE interface is `scipy.integrate.solve_ivp`, and numerical
integration can be done with `trapz`, `simps`, `quad`, `dblquad`, and `tplquad`.

Python updates since 2020
-------------------------
The core ideas above remain the same, but Python itself has added useful
features:

- `match`/`case` (3.10) for structured conditional logic.
- Dictionary merge and update with `|` and `|=` (3.9).
- F-string debug syntax `f"{var=}"` (3.8).
- Built-in generic type hints like `list[str]` (3.9) and union types
  `int | None` (3.10).
- `tomllib` (3.11) for reading TOML configuration files.
- Interpreter performance improvements in 3.11+ that make numeric code faster
  even before NumPy.

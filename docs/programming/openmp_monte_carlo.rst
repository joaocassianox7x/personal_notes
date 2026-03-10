OpenMP in C++: Monte Carlo Integral
===================================

Overview
--------
OpenMP is a lightweight way to parallelize loop-heavy C++ code with compiler pragmas.
A Monte Carlo integral is a good first example because each random sample is independent,
so the work splits cleanly across threads.

Problem setup
-------------
We estimate the quarter-circle integral

.. math::

   I = \int_0^1 \sqrt{1 - x^2}\,dx = \frac{\pi}{4}.

Instead of evaluating the function directly, we use a hit-or-miss Monte Carlo method:
sample points uniformly in the unit square and count how many satisfy

.. math::

   x^2 + y^2 \le 1.

If ``N`` points are sampled and ``H`` land inside the quarter circle, then

.. math::

   \hat{I} = \frac{H}{N},
   \qquad
   \hat{\pi} = 4\frac{H}{N}.

Parallel C++ implementation
---------------------------
The code below uses three OpenMP ideas:

- ``#pragma omp parallel reduction(+:hits)`` gives each thread a private counter and sums the result at the end.
- ``#pragma omp for schedule(static)`` splits the sample loop evenly across threads.
- Each thread owns its own RNG state, which avoids locks and contention.

.. literalinclude:: ../_static_files/codes/openmp_monte_carlo.cpp
   :language: cpp
   :caption: docs/_static_files/codes/openmp_monte_carlo.cpp

Monte Carlo animation
---------------------
The animation below shows the hit-or-miss estimator converging as more points are drawn.
Green points fall inside the quarter circle, orange points miss it, and the running
estimate approaches :math:`\pi`.

.. image:: ../_static_files/images/openmp_monte_carlo.gif
   :alt: Monte Carlo sampling inside a quarter circle for an OpenMP integral example
   :align: center
   :width: 80%

Benchmarks
----------
These measurements were collected locally on March 10, 2026 on an AMD Ryzen 3 4100
(4 cores / 8 threads) with GCC 12.3.1, ``-O3 -std=c++17 -fopenmp``, and
``10^9`` Monte Carlo samples.

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18

   * - Threads
     - Median time (s)
     - Speedup vs 1 thread
     - :math:`\hat{\pi}`
     - Absolute error
   * - 1
     - 3.583610
     - 1.00x
     - 3.141620
     - 0.000028
   * - 2
     - 1.863072
     - 1.92x
     - 3.141524
     - 0.000069
   * - 4
     - 1.388099
     - 2.58x
     - 3.141683
     - 0.000090

The 2-thread result is close to ideal scaling. At 4 threads the run is still clearly faster,
but the speedup becomes sublinear because the loop is no longer purely compute-bound:
thread startup, reduction overhead, and memory traffic start to matter.

Reproduce the example
---------------------
Compile the program:

.. code-block:: bash

   g++ -O3 -std=c++17 -fopenmp docs/_static_files/codes/openmp_monte_carlo.cpp -o /tmp/openmp_monte_carlo

Run the benchmark points used above:

.. code-block:: bash

   OMP_NUM_THREADS=1 /tmp/openmp_monte_carlo 1000000000
   OMP_NUM_THREADS=2 /tmp/openmp_monte_carlo 1000000000
   OMP_NUM_THREADS=4 /tmp/openmp_monte_carlo 1000000000

Regenerate the GIF:

.. code-block:: bash

   python docs/_static_files/codes/openmp_monte_carlo_assets.py

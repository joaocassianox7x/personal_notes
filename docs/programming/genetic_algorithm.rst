Genetic Algorithms
==================

Core Idea
---------

A genetic algorithm (GA) is a population-based optimization method inspired by
selection, reproduction, and mutation in biology.

Instead of improving a single solution at a time, a GA keeps a population

.. math::

   P^{(t)} = \{x_1^{(t)}, x_2^{(t)}, \dots, x_N^{(t)}\}

at generation :math:`t`, evaluates the quality of each candidate with a fitness
function :math:`f(x)`, and produces a new population by repeatedly applying:

- selection,
- crossover,
- mutation,
- elitism.

The main advantage is that GAs can explore complicated, non-convex, and even
discrete search spaces where gradient-based methods are unavailable or unreliable.


The Basic Loop
--------------

For a maximization problem, the standard genetic algorithm is:

1. Initialize a random population.
2. Evaluate fitness :math:`f(x_i)` for every individual.
3. Select better individuals with higher probability.
4. Recombine parents with crossover to create offspring.
5. Mutate offspring to maintain diversity.
6. Copy a few elite individuals unchanged.
7. Repeat until a stopping criterion is met.

In pseudocode:

.. code-block:: text

   initialize population P
   evaluate fitness on P

   while not stop:
       keep elites
       while new population not full:
           parent1, parent2 = selection(P)
           child1, child2 = crossover(parent1, parent2)
           child1 = mutation(child1)
           child2 = mutation(child2)
           add children
       P = new population
       evaluate fitness on P


Main Components
---------------

**Fitness**

The algorithm needs a scalar score:

.. math::

   f : \mathcal{X} \to \mathbb{R}

that measures how good a candidate is.

For minimization, a common practical trick is to optimize :math:`-L(x)` or use a
rank-based selection rule instead of raw values.

**Selection**

Selection gives better candidates more chances to reproduce.

Common strategies:

- roulette-wheel selection,
- rank selection,
- tournament selection.

In tournament selection, we sample a few individuals and keep the best among them.
This is simple, robust, and easy to implement.

**Crossover**

Crossover combines information from two parents to produce offspring. It is the
main exploitation mechanism of a GA.

**Mutation**

Mutation injects random variation so that the population does not collapse too
early around a bad local optimum.

**Elitism**

Elitism copies the best few individuals directly into the next generation. This
prevents losing the best solution found so far.


Method 1: Binary-Coded Genetic Algorithm
----------------------------------------

In a binary-coded GA, each candidate is represented by a bit string:

.. math::

   x \leftrightarrow b_1 b_2 \dots b_L, \qquad b_i \in \{0,1\}.

If the true variable is continuous on :math:`[a,b]`, the bit string is decoded
into an integer

.. math::

   u = \sum_{k=1}^{L} b_k 2^{L-k}

and then mapped to the interval:

.. math::

   x = a + \frac{u}{2^L - 1}(b-a).

This creates a discrete approximation of the continuous search space.

Binary operators
~~~~~~~~~~~~~~~~

**One-point crossover**

Choose a cut point and swap the tails of two parents.

**Bit-flip mutation**

For each bit, apply:

.. math::

   b_i' =
   \begin{cases}
   1-b_i, & \text{with probability } p_m \\
   b_i, & \text{otherwise}
   \end{cases}

Binary advantages
~~~~~~~~~~~~~~~~~

- Simple and classical formulation.
- Natural for symbolic or combinatorial variables.
- Easy to reason about with discrete building blocks.

Binary limitations
~~~~~~~~~~~~~~~~~~

- Continuous variables must be quantized.
- Small changes in the decoded value may require several bit changes.
- Resolution depends on chromosome length.


Method 2: Real-Coded Genetic Algorithm
--------------------------------------

In a real-coded GA, each candidate stores the actual continuous parameters:

.. math::

   x = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d.

There is no encoding/decoding step. This is usually a better fit for continuous
optimization.

Real-coded operators
~~~~~~~~~~~~~~~~~~~~

**Arithmetic crossover**

Given two parents :math:`p_1` and :math:`p_2`, generate offspring by interpolation:

.. math::

   c_1 = \alpha p_1 + (1-\alpha)p_2,
   \qquad
   c_2 = (1-\alpha)p_1 + \alpha p_2,

with :math:`\alpha \in [0,1]`.

**Gaussian mutation**

Perturb the child with small random noise:

.. math::

   x' = x + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2).

Real-coded advantages
~~~~~~~~~~~~~~~~~~~~~

- Natural for continuous parameter search.
- No quantization error.
- Usually converges more smoothly in real-valued domains.

Real-coded limitations
~~~~~~~~~~~~~~~~~~~~~~

- Operator design matters more.
- Poor mutation scale can make the search too random or too conservative.
- Constraints must often be handled explicitly by clipping, penalties, or repair.


Visual View of the Operators
----------------------------

.. image:: ../_static_files/images/genetic_algorithm_operators.png
   :alt: Binary-coded and real-coded genetic algorithm operators
   :align: center
   :width: 95%

The left panel shows the classical binary representation with one-point crossover
and bit-flip mutation. The right panel shows a real-coded variant using arithmetic
crossover followed by Gaussian mutation.


Toy Optimization Example
------------------------

To compare the two methods, we optimize a 1D multimodal objective. The exact
shape is not the point; what matters is that it has multiple local optima, so the
algorithm must balance exploration and exploitation.

The population starts spread across the full interval and progressively concentrates
around the best region discovered so far.

.. image:: ../_static_files/images/genetic_algorithm_landscape.png
   :alt: Binary and real-coded genetic algorithm populations moving on a multimodal function
   :align: center
   :width: 100%

The binary-coded GA can only place individuals on a discrete grid induced by the
bit representation. The real-coded GA can move continuously, which often makes
fine adjustment easier near a good optimum.


Convergence Behavior
--------------------

.. image:: ../_static_files/images/genetic_algorithm_convergence.png
   :alt: Convergence curves for binary-coded and real-coded genetic algorithms
   :align: center
   :width: 90%

The solid curves show the best fitness in each generation, and the dashed curves
show the population mean. In this toy example, both methods reach essentially the
same optimum, but the real-coded version typically refines the location more
smoothly because it is not limited by bit resolution.


Binary vs Real-Coded: When to Use Each
--------------------------------------

Use a **binary-coded GA** when:

- the search variables are naturally discrete,
- the problem is combinatorial,
- chromosome logic matters more than geometric distance in :math:`\mathbb{R}^d`.

Use a **real-coded GA** when:

- the parameters are continuous,
- you want smoother local refinement,
- decoding overhead is unnecessary.

In modern ML and optimization practice, real-coded variants are usually preferred
for continuous hyperparameters, controller tuning, and black-box objective search.


Where Genetic Algorithms Fit in Optimization and AI
---------------------------------------------------

Genetic algorithms are most useful when:

- the objective is non-differentiable,
- gradients are unavailable,
- the search space mixes discrete and continuous variables,
- the function is noisy or simulator-based,
- global exploration matters more than precise local convex optimization.

Common examples:

- hyperparameter search,
- neural architecture search,
- feature subset selection,
- scheduling and routing,
- control parameter tuning,
- symbolic rule discovery.


Common Pitfalls
---------------

**Premature convergence**

If selection pressure is too strong, the population becomes too similar too early.

**Too little mutation**

Without mutation, diversity collapses and the search may get stuck.

**Too much mutation**

If mutation is too aggressive, the algorithm behaves like random search.

**Poor representation**

The encoding must match the problem. A bad chromosome design makes good solutions
hard to inherit.

**Over-expensive fitness**

GAs may require many evaluations, so they can be expensive when each evaluation is
a full simulation or model training run.


Reproduce the Figures
---------------------

Run the generator script:

.. code-block:: bash

   python docs/_static_files/codes/genetic_algorithm.py

It writes the PNGs under ``docs/_static_files/images`` and prints the best solution
found by each GA variant.


References and Further Reading
------------------------------

- **Holland, J. H.** (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press. Foundational text introducing the genetic algorithm framework.
- **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley. Classical engineering-oriented treatment of selection, crossover, and mutation.
- **Michalewicz, Z.** (1996). *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer. Strong practical reference on operators and representations.
- **Deb, K.** (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley. Standard reference for evolutionary methods beyond single-objective search.

Personal Notes
==============

.. raw:: html

   <section class="landing-hero">
     <p class="landing-kicker">Math, DS/AI, Physics, and Programming</p>
     <p class="landing-lede">
       A compact notebook for worked derivations, numerical methods, programming references,
       and study material that is easy to reuse.
     </p>
     <div class="landing-actions">
       <a class="pn-button pn-button-primary" href="intro.html">Open the overview</a>
       <a class="pn-button" href="aboutme.html">About the author</a>
     </div>
   </section>

   <section class="landing-grid">
     <a class="landing-card" href="math/index.html">
       <p class="landing-card-tag">Math</p>
       <h2>Core Mathematics</h2>
       <p>Short theory summaries with direct problem-solving methods and worked examples.</p>
       <ul>
         <li>Functions and limits</li>
         <li>Integrals and common techniques</li>
         <li>Residue and Stokes theorems</li>
       </ul>
     </a>
     <a class="landing-card" href="ds_ai/index.html">
       <p class="landing-card-tag">DS/AI</p>
       <h2>Data Science and AI</h2>
       <p>Practical notes on models, metrics, training dynamics, and machine learning foundations.</p>
       <ul>
         <li>Regression and classification</li>
         <li>Bias, variance, and metrics</li>
         <li>PCA, backpropagation, and transformers</li>
       </ul>
     </a>
     <a class="landing-card" href="physics/index.html">
       <p class="landing-card-tag">Physics</p>
       <h2>Computational Physics</h2>
       <p>Simulation notes, numerical methods, and worked references from computational physics topics.</p>
       <ul>
         <li>PPPM and Ewald notes</li>
         <li>Double pendulum and oscillators</li>
         <li>Split-operator techniques</li>
       </ul>
     </a>
     <a class="landing-card" href="programming/index.html">
       <p class="landing-card-tag">Programming</p>
       <h2>Code and Parallelism</h2>
       <p>Language notes and practical examples for scientific computing and performance-oriented programming.</p>
       <ul>
         <li>Python foundations for numerics</li>
         <li>Genetic algorithms</li>
         <li>OpenMP with Monte Carlo integration</li>
       </ul>
     </a>
   </section>

.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   aboutme
   intro

.. toctree::
   :maxdepth: 1
   :caption: Math
   :hidden:

   math/functions
   math/limits
   math/integrals
   math/stokes_theorem
   math/residue_theorem

.. toctree::
   :maxdepth: 1
   :caption: DS/AI
   :hidden:

   ds_ai/linear_regression
   ds_ai/logistic_regression
   ds_ai/bias_variance
   ds_ai/metrics
   ds_ai/backpropagation
   ds_ai/pca
   ds_ai/regularization
   ds_ai/transformers

.. toctree::
   :maxdepth: 1
   :caption: Physics
   :hidden:

   physics/pppm_ewald
   physics/double_pendulum
   physics/computational_derivatives
   physics/force_fields
   physics/lammps
   physics/harmonic_oscillator
   physics/split_operator

.. toctree::
   :maxdepth: 1
   :caption: Programming
   :hidden:

   programming/python
   programming/genetic_algorithm
   programming/openmp_monte_carlo

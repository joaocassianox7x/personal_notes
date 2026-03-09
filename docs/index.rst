Personal Notes
==============

.. raw:: html

   <section class="landing-hero">
     <p class="landing-kicker">Physics, Math, and Machine Learning</p>
     <p class="landing-lede">
       A compact notebook for worked derivations, numerical methods, and study references.
       The goal is speed: short explanations, direct formulas, and examples that are easy to reuse.
     </p>
     <div class="landing-actions">
       <a class="pn-button pn-button-primary" href="intro.html">Open the overview</a>
       <a class="pn-button" href="aboutme.html">About the author</a>
     </div>
   </section>

   <section class="landing-grid">
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
     <a class="landing-card" href="math/index.html">
       <p class="landing-card-tag">Math</p>
       <h2>Core Mathematics</h2>
       <p>Short theory summaries with direct problem-solving methods and worked examples.</p>
       <ul>
         <li>Functions and limits</li>
         <li>Integrals and common techniques</li>
         <li>Residue theorem</li>
       </ul>
     </a>
     <a class="landing-card" href="ds_ai/index.html">
       <p class="landing-card-tag">Data</p>
       <h2>DS, ML, and AI</h2>
       <p>Practical notes on models, metrics, training dynamics, and machine learning foundations.</p>
       <ul>
         <li>Regression and classification</li>
         <li>Bias, variance, and metrics</li>
         <li>PCA, backpropagation, transformers</li>
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
   :caption: Physics and Computational Physics
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
   :caption: Math
   :hidden:

   math/functions
   math/limits
   math/integrals
   math/residue_theorem

.. toctree::
   :maxdepth: 1
   :caption: DS, ML, and AI
   :hidden:

   ds_ai/python
   ds_ai/linear_regression
   ds_ai/logistic_regression
   ds_ai/bias_variance
   ds_ai/metrics
   ds_ai/backpropagation
   ds_ai/pca
   ds_ai/transformers

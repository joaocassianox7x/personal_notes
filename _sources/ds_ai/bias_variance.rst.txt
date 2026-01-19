Bias - Variance Decomposition
======================

Prediction error can be decomposed into three irreducible components: bias, variance, and noise. Understanding this decomposition is essential for:

- Diagnosing why a model performs poorly
- Selecting appropriate model complexity
- Understanding overfitting and underfitting
- Designing effective regularization strategies

Expected Prediction Error
--------------------------

Consider a regression problem where we have:

.. math::

    Y = f(X) + \epsilon

where:

- :math:`Y` is the target variable
- :math:`X` is the input feature vector
- :math:`f(X)` is the unknown true function
- :math:`\epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)` is irreducible noise

**Goal:** Estimate :math:`f(X)` with a learned function :math:`\hat{f}(X)` using training data :math:`\mathcal{D}`.

Expected Squared Error
-----------------------

For a test point :math:`(X_0, Y_0)`, the expected squared error is:

.. math::

    \text{EPE}(X_0) = \mathbb{E}_{Y_0, \mathcal{D}} \left[ (Y_0 - \hat{f}(X_0))^2 \right]

**Step 1: Expand the squared error**

.. math::

    \text{EPE}(X_0) = \mathbb{E}_{Y_0, \mathcal{D}} \left[ Y_0^2 - 2Y_0 \hat{f}(X_0) + \hat{f}(X_0)^2 \right]

**Step 2: Substitute the true model**

Since :math:`Y_0 = f(X_0) + \epsilon`:

.. math::

    \text{EPE}(X_0) = \mathbb{E}_{Y_0, \mathcal{D}} \left[ (f(X_0) + \epsilon)^2 - 2(f(X_0) + \epsilon)\hat{f}(X_0) + \hat{f}(X_0)^2 \right]

**Step 3: Expand and separate terms**

.. math::

    \text{EPE}(X_0) = \mathbb{E}[f(X_0)^2] + \mathbb{E}[\epsilon^2] + 2\mathbb{E}[f(X_0)\epsilon] 
    - 2\mathbb{E}[f(X_0)\hat{f}(X_0)] - 2\mathbb{E}[\epsilon \hat{f}(X_0)] + \mathbb{E}[\hat{f}(X_0)^2]

**Step 4: Simplify using independence**

Since :math:`\epsilon` is independent of both :math:`f(X_0)` and :math:`\hat{f}(X_0)`, and :math:`\mathbb{E}[\epsilon] = 0`:

.. math::

    \text{EPE}(X_0) = f(X_0)^2 + \sigma_\epsilon^2 - 2f(X_0)\mathbb{E}_\mathcal{D}[\hat{f}(X_0)] + \mathbb{E}_\mathcal{D}[\hat{f}(X_0)^2]

**Step 5: Complete the decomposition**

Add and subtract :math:`(\mathbb{E}_\mathcal{D}[\hat{f}(X_0)])^2`:

.. math::

    \text{EPE}(X_0) = \left[ f(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)] \right]^2 + \mathbb{E}_\mathcal{D}[\hat{f}(X_0)^2] - (\mathbb{E}_\mathcal{D}[\hat{f}(X_0)])^2 + \sigma_\epsilon^2

The Fundamental Bias-Variance Decomposition
---------------------------------------------

The above derivation yields the fundamental decomposition:

.. math::

    \text{EPE}(X_0) = \underbrace{\left[ f(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)] \right]^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_\mathcal{D}\left[ (\hat{f}(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)])^2 \right]}_{\text{Variance}} + \underbrace{\sigma_\epsilon^2}_{\text{Irreducible Error}}

**Bias Term:**

.. math::

    \text{Bias}(X_0) = f(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)]

Represents the systematic error due to model assumptions. A model with high bias makes strong assumptions about the data (e.g., linear when data is nonlinear).

**Variance Term:**

.. math::

    \text{Var}(X_0) = \mathbb{E}_\mathcal{D}\left[ (\hat{f}(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)])^2 \right]

Represents the sensitivity of the model to fluctuations in the training data. High variance means small changes in training data lead to large changes in predictions.

**Irreducible Error (Noise):**

.. math::

    \text{Noise} = \sigma_\epsilon^2

The inherent randomness in the target variable that cannot be explained by features, regardless of model quality.

Expected Test Error (Integrated)
---------------------------------

Averaging over the entire test set:

.. math::

    \text{Test Error} = \mathbb{E}_{X_0} \left[ \text{EPE}(X_0) \right] = \mathbb{E}_{X_0}[\text{Bias}^2(X_0)] + \mathbb{E}_{X_0}[\text{Var}(X_0)] + \sigma_\epsilon^2

This is the **total expected test error** that we aim to minimize.

Detailed Analysis of Bias and Variance
--------------------------

Bias: Definition and Interpretation
------------------------------------

**Mathematical Definition:**

.. math::

    B(X_0) = f(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)]

**Interpretation:**

- Bias measures how far the *average* prediction (over all possible training sets) is from the true value
- It reflects the **systematic underfitting** of the model
- Caused by model architecture being too simple to capture true function

**Common Sources of Bias:**

1. **Linear models on nonlinear data:**
   
   True: :math:`f(X) = X^2 + \sin(X)`
   
   Model: :math:`\hat{f}(X) = \beta_0 + \beta_1 X`
   
   The model cannot capture the nonlinearity, creating high bias.

2. **Missing relevant features:**
   
   .. math::
   
       Y = f(X_1, X_2) + \epsilon
   
   If we only use :math:`X_1`, we introduce omitted variable bias.

3. **Over-regularization:**
   
   Heavy regularization (large :math:`\lambda` in :math:`L = \text{MSE} + \lambda \|w\|^2`) pushes coefficients toward zero, creating bias.

Variance: Definition and Interpretation
----------------------------------------

**Mathematical Definition:**

.. math::

    V(X_0) = \mathbb{E}_\mathcal{D}\left[ (\hat{f}(X_0) - \mathbb{E}_\mathcal{D}[\hat{f}(X_0)])^2 \right]

**Alternative form (law of total variance):**

.. math::

    V(X_0) = \mathbb{E}_\mathcal{D}[\hat{f}(X_0)^2] - (\mathbb{E}_\mathcal{D}[\hat{f}(X_0)])^2

**Interpretation:**

- Variance measures how much the predictions vary across different training sets
- It reflects **overfitting**: model fits training noise rather than true signal
- High variance indicates the model is too sensitive to training data specifics

**Common Sources of Variance:**

1. **Complex models with insufficient data:**
   
   A 10-degree polynomial fit to 20 data points will vary wildly with small data changes.

2. **Small training set size:**
   
   With :math:`N` samples, parameter estimates become more unstable.

3. **Under-regularization:**
   
   Small :math:`\lambda` allows model to fit training noise.

4. **High-dimensional feature spaces:**
   
   More parameters to learn with same amount of data → higher variance.

Relationship Between Bias and Variance
---------------------------------------

**Key Trade-off:**

- **Simple models:** Low variance (stable across datasets), High bias (miss true pattern)
- **Complex models:** Low bias (can capture patterns), High variance (sensitive to noise)

**Mathematical insight:**

As model complexity increases:

.. math::

    \text{Bias}^2 \text{ decreases (monotonically)}

.. math::

    \text{Variance} \text{ increases (monotonically)}

**Total error:**

.. math::

    \text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}

The optimal model complexity minimizes total error, typically at an intermediate complexity level.

Bias-Variance in Different Contexts
--------------------------

Linear Regression
------------------

For linear regression :math:`\hat{f}(X) = X\beta`, with true model :math:`Y = f(X) + \epsilon`:

**Bias of OLS estimator:**

.. math::

    \text{Bias}(X_0) = X_0(\beta^* - \mathbb{E}[\hat{\beta}])

If the true model is **linear** (:math:`f(X) = X\beta^*`), OLS is unbiased:

.. math::

    \mathbb{E}[\hat{\beta}] = \beta^*

So :math:`\text{Bias} = 0` when model is correctly specified.

**Variance of OLS estimator:**

.. math::

    \text{Var}(\hat{f}(X_0)) = X_0 \text{Cov}(\hat{\beta}) X_0^T = X_0 (\sigma_\epsilon^2 (X^TX)^{-1}) X_0^T

Increases with:

- Noise variance :math:`\sigma_\epsilon^2`
- High multicollinearity in :math:`X` (large :math:`(X^TX)^{-1}`)
- More features (larger :math:`(X^TX)^{-1}`)

Regularized Regression
----------------------

**Ridge Regression** adds penalty: :math:`\min_\beta \|Y - X\beta\|^2 + \lambda \|\beta\|^2`

The estimator becomes:

.. math::

    \hat{\beta}_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^T Y

**Bias introduced by regularization:**

.. math::

    \text{Bias}(\hat{\beta}_{\text{ridge}}) = -\lambda (X^TX + \lambda I)^{-1} X^T X \beta^*

As :math:`\lambda \to 0`: Bias :math:`\to 0`, but Variance :math:`\to \infty`

As :math:`\lambda \to \infty`: Bias :math:`\to -X\beta^*`, but Variance :math:`\to 0`

**Optimal regularization** balances the two.

Decision Trees and Ensemble Methods
------------------------------------

**Decision Trees:**

- **Low bias:** Can approximate any function given enough depth
- **High variance:** Small training perturbations cause large tree changes
- Prone to overfitting on noisy data

**Random Forests (Bagging):**

Ensemble of :math:`B` bootstrap samples, each with tree :math:`\hat{f}_b(X)`:

.. math::

    \hat{f}_{\text{bag}}(X) = \frac{1}{B} \sum_{b=1}^B \hat{f}_b(X)

**Variance reduction:**

.. math::

    \text{Var}(\hat{f}_{\text{bag}}) = \frac{\text{Var}(\hat{f})}{B} + \frac{B-1}{B} \text{Cov}(\hat{f}_i, \hat{f}_j)

If trees are not perfectly correlated, variance drops significantly.

Bias remains approximately unchanged (still low, from base tree).

**Boosting:**

Sequentially builds weak learners, reducing both bias and variance:

.. math::

    \hat{f}_{\text{boost}}(X) = \sum_{b=1}^B \alpha_b \hat{f}_b(X)

Each iteration focuses on residuals from previous iteration, progressively reducing bias.

.. image:: ../_static_files/images/bias_variance_tradeoff.png
   :alt: Bias-Variance tradeoff illustration showing total error, bias squared, and variance
   :width: 100%
   :align: center

Cross-Validation and Bias-Variance
--------------------------

Leave-One-Out Cross-Validation (LOOCV)
---------------------------------------

LOOCV error estimates:

.. math::

    \text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat{f}^{(-i)}(X_i))^2

where :math:`\hat{f}^{(-i)}` is trained on all data except observation :math:`i`.

**Property:** Nearly unbiased estimator of test error (high-variance estimate, though).

K-Fold Cross-Validation
------------------------

Splits data into :math:`k` folds:

.. math::

    \text{CV}_{(k)} = \frac{1}{k} \sum_{j=1}^k \text{MSE}_j

where :math:`\text{MSE}_j` is error on fold :math:`j`.

**Tradeoff:**

- Larger :math:`k` (e.g., :math:`k=n`, LOOCV): Lower bias, higher variance
- Smaller :math:`k` (e.g., :math:`k=5`): Higher bias, lower variance
- Standard choice: :math:`k=5` or :math:`k=10` balances both

.. image:: ../_static_files/images/cross_validation_bias_variance.png
   :alt: Cross-validation error curves and quality metrics
   :width: 100%
   :align: center

Learning Curves: Diagnosing Bias-Variance Problems
--------------------------

Learning curves plot training and validation error versus training set size :math:`n`.

**Underfitting (High Bias):**

.. math::

    \lim_{n \to \infty} \text{Train Error} = \lim_{n \to \infty} \text{Val Error} = \text{High}

Both curves plateau at a high error level, with small gap between them.

**Overfitting (High Variance):**

.. math::

    \text{Train Error} \ll \text{Val Error} \text{ for all } n

Large gap persists even with more data; training error stays low while validation error remains high.

**Good fit:**

.. math::

    \text{Train Error} \approx \text{Val Error} \approx \text{Bayes Error}

Both curves converge with manageable gap.

.. image:: ../_static_files/images/learning_curves_bias_variance.png
   :alt: Learning curves showing underfitting, good fit, and overfitting scenarios
   :width: 100%
   :align: center

Model Complexity Examples
--------------------------

Below are examples of fitted models with varying polynomial degrees showing the bias-variance tradeoff:

.. image:: ../_static_files/images/model_complexity_effect.png
   :alt: Fitted polynomial models of varying degrees demonstrating bias-variance
   :width: 100%
   :align: center

Practical Strategies to Control Bias-Variance
--------------------------

Reducing Bias (Model is Too Simple)
-----------------------------------

1. **Increase model complexity:**
   
   - Use higher-degree polynomials
   - Add more features (with caution)
   - Use nonlinear models (neural networks, kernels)

2. **Reduce regularization:**
   
   - Decrease :math:`\lambda` in ridge/LASSO
   - Reduce tree depth constraints
   - Increase ensemble size

3. **Feature engineering:**
   
   - Add interaction terms
   - Add polynomial features
   - Domain-specific transformations

**Example:** If polynomial regression underfits, increase degree from 2 to 3 or 5.

Reducing Variance (Model is Too Complex)
-----------------------------------------

1. **Decrease model complexity:**
   
   - Use lower-degree polynomials
   - Reduce number of features (feature selection)
   - Limit tree depth, require minimum samples per leaf

2. **Increase regularization:**
   
   - Increase :math:`\lambda` in ridge/LASSO
   - Add early stopping in boosting/neural networks
   - Use dropout in neural networks

3. **Get more training data:**
   
   .. math::
   
       \text{Var} \propto \frac{\sigma_\epsilon^2}{n}
   
   Increasing :math:`n` reduces variance directly.

4. **Ensemble methods:**
   
   - Bagging/Random Forests reduce variance without increasing bias
   - Helps decorrelate predictions across bootstrap samples

5. **Hyperparameter tuning:**
   
   Use cross-validation to find :math:`\lambda`, :math:`k`, etc. that minimize validation error.

.. image:: ../_static_files/images/bias_variance_reduction_strategies.png
   :alt: Strategies for reducing bias and variance in models
   :width: 100%
   :align: center

The Bayes Error: Ultimate Lower Bound
--------------------------

**Bayes error** (also **irreducible error**):

.. math::

    \epsilon_{\text{Bayes}} = \inf_{\hat{f}} \mathbb{E}[(Y - \hat{f}(X))^2]

This is the best possible error using *any* model, achieved when :math:`\hat{f}(X) = \mathbb{E}[Y|X]`.

**Total error decomposition:**

.. math::

    \text{Total Error} = \underbrace{\text{Bias}^2 + \text{Variance}}_{\text{Reducible Error}} + \underbrace{\epsilon_{\text{Bayes}}}_{\text{Irreducible Error}}

In noisy problems (:math:`\sigma_\epsilon^2` large), even perfect models incur significant error.

**Practical consequence:**

- Don't obsess over achieving zero training error if noise is high
- Target validation error near Bayes error + small margin for model imperfection

Bias-Variance in Classification
--------------------------

For classification with 0-1 loss:

.. math::

    L(Y, \hat{f}(X)) = \mathbb{1}[Y \neq \hat{f}(X)]

**Misclassification error:**

.. math::

    \text{Err}(X_0) = \Pr(\hat{f}(X_0) \neq f(X_0))

Bias-variance decomposition still applies but is more nuanced:

.. math::

    \text{Bias} = \Pr(f(X_0) \neq \mathbb{E}[\hat{f}(X_0)])

Models with high variance may still have **low bias** in classification if variance is around the correct decision boundary.

**Example:** A 3-nearest neighbor classifier has higher variance but potentially lower bias than 1-nearest neighbor.

.. image:: ../_static_files/images/bias_variance_classification.png
   :alt: Bias-variance tradeoff in classification with decision boundaries
   :width: 100%
   :align: center

Summary Table
--------------------------

.. list-table:: Bias-Variance Characteristics
   :widths: 20, 25, 25, 30
   :header-rows: 1

   * - Aspect
     - **High Bias**
     - **High Variance**
     - **Balanced**
   * - Model Type
     - Too simple (linear on nonlinear)
     - Too complex (overfitting)
     - Appropriate complexity
   * - Training Error
     - High
     - Low
     - Moderate
   * - Validation Error
     - High
     - High
     - Low
   * - Train-Val Gap
     - Small
     - Large
     - Small
   * - Cause
     - Underfitting
     - Overfitting
     - Good fit
   * - Cure
     - Add complexity, features, reduce :math:`\lambda`
     - Reduce complexity, increase :math:`\lambda`, get data
     - Fine-tune hyperparameters
   * - Learning Curve
     - Both curves high, converge
     - Large gap, gap persists
     - Small gap, both converge to low error


References and Further Reading
------------------------------
- **Geman, S., Bienenstock, E., & Doursat, R.** (1992). Neural networks and the bias/variance dilemma. *Neural Computation*, 4(1), 1–58. Classic paper formalizing the tradeoff for neural nets.
- **Domingos, P.** (2000). A unified bias-variance decomposition. In *ICML*. Extends the decomposition to many loss functions beyond squared error.
- **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 7 provides practical guidance on bias-variance and model selection.
- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. Sections 1.3 and 3.2 connect bias-variance with Bayesian viewpoints.
- **Kuhn, M., & Johnson, K.** (2013). *Applied Predictive Modeling*. Springer. Chapter 4 covers bias-variance diagnostics with real-world examples.

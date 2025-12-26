Linear Regression
=================

Core idea
---------
Fit a linear relationship between features and a target: :math:`y \approx X\beta + \varepsilon`.

Model
-----
- Hypothesis: :math:`\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j` (can include interaction or polynomial terms).
- Matrix form: :math:`\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}`.

Objective
---------
Minimize squared error:

.. math::

   \min_{\boldsymbol{\beta}} \left\| \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right\|_2^2.

- Closed form (when :math:`\mathbf{X}^\top\mathbf{X}` is invertible):

  .. math::

     \hat{\boldsymbol{\beta}} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1}\mathbf{X}^\top \mathbf{y}.

- With :math:`L_2` regularization (Ridge):

  .. math::

     \hat{\boldsymbol{\beta}} = \left(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}\right)^{-1}\mathbf{X}^\top \mathbf{y}.

Classical assumptions
---------------------
- Linearity of the relationship between predictors and target.
- Independent errors.
- Homoscedasticity (constant error variance).
- No multicollinearity among predictors.
- Errors are normally distributed with mean :math:`0` and variance :math:`\sigma^2` (for inference).

Diagnostics
-----------
- :math:`R^2` and adjusted :math:`R^2`.
- Residual plots for heteroscedasticity or nonlinearity.
- Variance Inflation Factor (VIF) for multicollinearity.
- Train/validation split or cross-validation to assess generalization.

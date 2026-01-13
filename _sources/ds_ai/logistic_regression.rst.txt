Logistic Regression
===================

Overview
--------
Binary classifier modeling :math:`p(y=1\mid x) = \sigma(w^T x + b)` with logistic link :math:`\sigma(t)=1/(1+e^{-t})`. Parameters are fitted by maximizing likelihood or minimizing binary cross-entropy.

Model and loss
--------------
.. math::

   \hat{y} = \sigma(X w + b), \qquad
   \mathcal{L}(w,b) = -\frac{1}{m} \sum_{i=1}^m \big[y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)\big].

Gradient (vectorized)
---------------------
.. math::

   \nabla_w \mathcal{L} = \frac{1}{m} X^T (\hat{y} - y), \qquad
   \partial_b \mathcal{L} = \frac{1}{m} \mathbf{1}^T (\hat{y} - y).

Numpy implementation (batch GD)
-------------------------------
.. code-block:: python

   import numpy as np

   def sigmoid(z):
       return 1.0 / (1.0 + np.exp(-z))

   def logistic_regression_fit(X, y, lr=0.1, epochs=500, l2=0.0):
       m, n = X.shape
       w = np.zeros(n)
       b = 0.0
       for _ in range(epochs):
           z = X @ w + b
           yhat = sigmoid(z)
           error = yhat - y
           grad_w = (X.T @ error) / m + l2 * w
           grad_b = error.mean()
           w -= lr * grad_w
           b -= lr * grad_b
       return w, b

   def predict_proba(X, w, b):
       return sigmoid(X @ w + b)

   def predict(X, w, b, threshold=0.5):
       return (predict_proba(X, w, b) >= threshold).astype(int)

Notes
-----
- Feature scaling (zero mean, unit variance) improves convergence.
- L2 term (ridge) mitigates overfitting and multicollinearity.
- Decision boundary in 2D: :math:`w_0 x_0 + w_1 x_1 + b = 0` (line); in higher dimensions, a hyperplane.

Figures
-------
.. figure:: ../_static_files/images/logistic_regression_boundary.svg
   :alt: Logistic regression decision boundary and probabilities
   :align: center
   :figwidth: 65%

   Sigmoid-shaped probability contours with a linear decision boundary.

"""
Bias-Variance Tradeoff - Visualization and Analysis

This script generates comprehensive visualizations for understanding:
1. Bias-Variance decomposition
2. Learning curves under different scenarios
3. Effect of model complexity on bias and variance
4. Classification decision boundaries
5. Regularization parameter effects
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import erfc
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def generate_synthetic_data(n_samples, noise_level, seed=42):
    """Generate synthetic data for regression tasks."""
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_samples)
    # True function: cubic with some oscillations
    Y_true = 0.5 * X**3 - 2*X + 1 + 0.3*np.sin(2*X)
    Y = Y_true + np.random.normal(0, noise_level, n_samples)
    return X, Y, Y_true


def plot_bias_variance_tradeoff():
    """
    Plot the classic bias-variance tradeoff curve showing total error decomposition.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Model complexity axis
    complexity = np.linspace(0, 100, 500)
    
    # Theoretical curves (approximated)
    bias_squared = 1.0 / (1.0 + complexity / 20.0)  # Decreases with complexity
    variance = 0.01 + 0.003 * complexity  # Increases with complexity
    noise = 0.02 * np.ones_like(complexity)  # Constant irreducible error
    total_error = bias_squared + variance + noise
    
    # Plot components
    ax.plot(complexity, bias_squared, 'b-', linewidth=3, label='Bias²')
    ax.plot(complexity, variance, 'r-', linewidth=3, label='Variance')
    ax.axhline(y=0.02, color='green', linewidth=2.5, linestyle='--', label='Irreducible Error (noise)')
    ax.plot(complexity, total_error, 'k-', linewidth=3.5, label='Total Error')
    
    # Mark optimal complexity
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]
    ax.plot(optimal_complexity, optimal_error, 'go', markersize=12, markeredgewidth=2, 
            markeredgecolor='darkgreen', label='Optimal Complexity')
    ax.axvline(x=optimal_complexity, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Annotations
    ax.annotate(f'Optimal\n({optimal_complexity:.1f}, {optimal_error:.3f})',
                xy=(optimal_complexity, optimal_error), xytext=(optimal_complexity+15, optimal_error+0.15),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))
    
    # Regions
    ax.axvspan(0, optimal_complexity, alpha=0.05, color='blue', label='Underfitting Region')
    ax.axvspan(optimal_complexity, 100, alpha=0.05, color='red', label='Overfitting Region')
    
    ax.set_xlabel('Model Complexity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title(r'Bias-Variance Tradeoff: Total Error Decomposition', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 0.6])
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: bias_variance_tradeoff.png")
    plt.close()


def plot_learning_curves():
    """
    Plot learning curves showing training and validation error versus training set size
    for underfitting, good fit, and overfitting scenarios.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    train_sizes = np.array([5, 10, 15, 20, 30, 50, 75, 100, 150, 200])
    
    # Scenario 1: Underfitting (High Bias)
    ax = axes[0]
    train_error_underfitting = 0.3 + 0.05 / np.sqrt(train_sizes)
    val_error_underfitting = 0.32 + 0.03 / np.sqrt(train_sizes)
    
    ax.plot(train_sizes, train_error_underfitting, 'b-o', linewidth=2.5, markersize=8, label='Training Error')
    ax.plot(train_sizes, val_error_underfitting, 'r-s', linewidth=2.5, markersize=8, label='Validation Error')
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Bayes Error')
    
    ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Underfitting\n(High Bias)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.5])
    
    # Scenario 2: Good Fit
    ax = axes[1]
    train_error_good = 0.08 + 0.03 / np.sqrt(train_sizes)
    val_error_good = 0.1 + 0.05 / np.sqrt(train_sizes)
    
    ax.plot(train_sizes, train_error_good, 'b-o', linewidth=2.5, markersize=8, label='Training Error')
    ax.plot(train_sizes, val_error_good, 'r-s', linewidth=2.5, markersize=8, label='Validation Error')
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Bayes Error')
    
    ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Good Fit\n(Balanced)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.5])
    
    # Scenario 3: Overfitting (High Variance)
    ax = axes[2]
    train_error_overfitting = 0.02 + 0.01 / train_sizes
    val_error_overfitting = 0.25 - 0.08 / np.sqrt(train_sizes)
    
    ax.plot(train_sizes, train_error_overfitting, 'b-o', linewidth=2.5, markersize=8, label='Training Error')
    ax.plot(train_sizes, val_error_overfitting, 'r-s', linewidth=2.5, markersize=8, label='Validation Error')
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Bayes Error')
    
    ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Overfitting\n(High Variance)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'learning_curves_bias_variance.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: learning_curves_bias_variance.png")
    plt.close()


def plot_model_complexity_effect():
    """
    Visualize fitted models with varying complexity showing bias-variance tradeoff.
    """
    # Generate true function and data
    X = np.linspace(-3, 3, 100)
    X_plot = np.linspace(-3.5, 3.5, 300)
    
    # True function
    Y_true = 0.5 * X_plot**3 - 2*X_plot + 1 + 0.3*np.sin(2*X_plot)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    np.random.seed(42)
    Y = 0.5 * X**3 - 2*X + 1 + 0.3*np.sin(2*X) + np.random.normal(0, 0.5, len(X))
    
    degrees = [1, 2, 3, 5, 8, 12]
    titles = ['Degree 1\n(High Bias)', 'Degree 2\n(High Bias)', 'Degree 3\n(Good Fit)', 
              'Degree 5\n(Good Fit)', 'Degree 8\n(High Variance)', 'Degree 12\n(High Variance)']
    
    for idx, (degree, title) in enumerate(zip(degrees, titles)):
        ax = axes[idx // 3, idx % 3]
        
        # Fit polynomial
        coeffs = np.polyfit(X, Y, degree)
        poly = np.poly1d(coeffs)
        Y_fit = poly(X_plot)
        
        # Plot
        ax.scatter(X, Y, s=40, alpha=0.6, color='blue', label='Training Data', edgecolors='darkblue')
        ax.plot(X_plot, Y_true, 'g-', linewidth=2.5, label='True Function', alpha=0.8)
        ax.plot(X_plot, Y_fit, 'r-', linewidth=2.5, label='Fitted Model', alpha=0.8)
        
        # Shaded regions
        if degree <= 2:
            ax.text(0, 7, 'UNDERFITTING\n(High Bias)', fontsize=10, fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif degree >= 8:
            ax.text(0, 7, 'OVERFITTING\n(High Variance)', fontsize=10, fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        else:
            ax.text(0, 7, 'GOOD FIT\n(Balanced)', fontsize=10, fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-2, 8])
        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'model_complexity_effect.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: model_complexity_effect.png")
    plt.close()


def plot_bias_variance_reduction_strategies():
    """
    Visualize the effect of different strategies on bias and variance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # Top-left: Effect of regularization (lambda)
    ax = axes[0, 0]
    lambda_vals = np.logspace(-3, 2, 50)
    bias_ridge = np.sqrt(np.linspace(0.001, 0.3, 50))  # Increases with lambda
    var_ridge = 0.3 * np.exp(-lambda_vals)  # Decreases with lambda
    total_ridge = bias_ridge + var_ridge
    
    ax.semilogx(lambda_vals, bias_ridge, 'b-', linewidth=2.5, label='Bias (increases with λ)')
    ax.semilogx(lambda_vals, var_ridge, 'r-', linewidth=2.5, label='Variance (decreases with λ)')
    ax.semilogx(lambda_vals, total_ridge, 'k-', linewidth=3, label='Total Error')
    ax.set_xlabel('Regularization Parameter (λ)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Effect of Regularization\n(Ridge Regression)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    
    # Top-right: Effect of training set size
    ax = axes[0, 1]
    n_samples = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
    bias_n = 0.1 * np.ones_like(n_samples)  # Constant
    var_n = 0.2 / np.sqrt(n_samples)  # Decreases with n
    total_n = bias_n + var_n
    
    ax.loglog(n_samples, bias_n, 'b-o', linewidth=2.5, markersize=8, label='Bias (constant)')
    ax.loglog(n_samples, var_n, 'r-s', linewidth=2.5, markersize=8, label='Variance (∝ 1/√n)')
    ax.loglog(n_samples, total_n, 'k-^', linewidth=3, markersize=8, label='Total Error')
    ax.set_xlabel('Training Set Size (n)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Effect of Training Data Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    
    # Bottom-left: Effect of model complexity
    ax = axes[1, 0]
    complexity = np.linspace(0, 50, 100)
    bias_complex = 1.0 / (1.0 + complexity / 10.0)
    var_complex = 0.01 + 0.002 * complexity
    total_complex = bias_complex + var_complex
    
    ax.plot(complexity, bias_complex, 'b-', linewidth=2.5, label='Bias')
    ax.plot(complexity, var_complex, 'r-', linewidth=2.5, label='Variance')
    ax.plot(complexity, total_complex, 'k-', linewidth=3, label='Total Error')
    ax.set_xlabel('Model Complexity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Effect of Model Complexity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Ensemble vs Single Model
    ax = axes[1, 1]
    ensemble_sizes = np.array([1, 2, 5, 10, 20, 50, 100])
    var_single = 0.1 * np.ones_like(ensemble_sizes)
    var_ensemble = 0.1 / np.sqrt(ensemble_sizes)  # Reduced by averaging
    bias_all = 0.02 * np.ones_like(ensemble_sizes)  # Unchanged
    
    ax.loglog(ensemble_sizes, var_single, 'r-o', linewidth=2.5, markersize=8, label='Single Model Variance')
    ax.loglog(ensemble_sizes, var_ensemble, 'g-s', linewidth=2.5, markersize=8, label='Ensemble Variance')
    ax.loglog(ensemble_sizes, bias_all, 'b--', linewidth=2, label='Bias (unchanged)')
    ax.set_xlabel('Ensemble Size (B)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('Effect of Ensemble Methods\n(Bagging/Random Forests)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'bias_variance_reduction_strategies.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: bias_variance_reduction_strategies.png")
    plt.close()


def plot_bias_variance_classification():
    """
    Visualize bias-variance tradeoff in classification with decision boundaries.
    """
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Generate classification data
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Define classifiers with varying complexity
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    classifiers = [
        (KNeighborsClassifier(n_neighbors=11), 'KNN (k=11)\nHigh Bias'),
        (KNeighborsClassifier(n_neighbors=5), 'KNN (k=5)\nBalanced'),
        (KNeighborsClassifier(n_neighbors=1), 'KNN (k=1)\nHigh Variance'),
        (DecisionTreeClassifier(max_depth=1), 'Tree (depth=1)\nHigh Bias'),
        (DecisionTreeClassifier(max_depth=5), 'Tree (depth=5)\nBalanced'),
        (DecisionTreeClassifier(max_depth=20), 'Tree (depth=20)\nHigh Variance'),
    ]
    
    for idx, (clf, title) in enumerate(classifiers):
        ax = axes[idx // 3, idx % 3]
        
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, levels=np.array([0, 0.5, 1]), colors=['lightblue', 'lightcoral'])
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, levels=[0.5])
        
        # Data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, 
                            edgecolors='black', linewidths=0.5)
        
        # Training accuracy
        train_acc = clf.score(X, y)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Feature 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=10, fontweight='bold')
        ax.set_title(f'{title}\nAccuracy: {train_acc:.2%}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'bias_variance_classification.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: bias_variance_classification.png")
    plt.close()


def plot_cv_variance_bias():
    """
    Visualize cross-validation error curves for different folds and their variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left plot: Cross-validation curves for different k-fold values
    ax = axes[0]
    k_values = [2, 5, 10]
    alpha_vals = np.linspace(0.01, 1, 50)
    
    for k in k_values:
        # Simulated CV error (typically U-shaped)
        cv_error = 0.1 + (alpha_vals - 0.5)**2 * 0.3
        # Add noise scaled by k (larger k has lower variance in CV estimate)
        noise = np.random.normal(0, 0.02 / np.sqrt(k), len(alpha_vals))
        cv_error_noisy = cv_error + noise
        
        ax.plot(alpha_vals, cv_error_noisy, 'o-', linewidth=2, markersize=6, 
               label=f'{k}-Fold CV', alpha=0.7)
    
    ax.set_xlabel('Regularization Parameter (alpha)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cross-Validation Error', fontsize=11, fontweight='bold')
    ax.set_title('k-Fold Cross-Validation Error Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Bias-variance of CV error estimate
    ax = axes[1]
    k_fold_values = np.array([2, 3, 5, 10, 20])
    
    # Bias and variance of CV error estimate
    cv_bias = 0.01 * np.ones_like(k_fold_values)  # Small, roughly constant
    cv_variance = 0.05 / np.sqrt(k_fold_values)  # Decreases with k
    cv_mse = cv_bias**2 + cv_variance
    
    ax.plot(k_fold_values, cv_bias, 'b-o', linewidth=2.5, markersize=8, label='Bias of CV estimate')
    ax.plot(k_fold_values, cv_variance, 'r-s', linewidth=2.5, markersize=8, label='Variance of CV estimate')
    ax.plot(k_fold_values, cv_mse, 'k-^', linewidth=3, markersize=8, label='MSE of CV estimate')
    
    ax.set_xlabel('Number of Folds (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax.set_title('CV Estimate Quality vs k', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'cross_validation_bias_variance.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: cross_validation_bias_variance.png")
    plt.close()


def main():
    """Generate all bias-variance visualizations."""
    print("\n" + "="*60)
    print("  Bias-Variance Tradeoff - Generating Visualizations")
    print("="*60 + "\n")
    
    print(f"Output directory: {IMAGES_DIR}\n")
    
    print("Generating plots...")
    plot_bias_variance_tradeoff()
    plot_learning_curves()
    plot_model_complexity_effect()
    plot_bias_variance_reduction_strategies()
    plot_bias_variance_classification()
    plot_cv_variance_bias()
    
    print("\n" + "="*60)
    print("  ✓ All visualizations generated successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

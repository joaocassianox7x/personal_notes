#!/usr/bin/env python3
"""Generate figures for L1 vs L2 regularization page."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

OUT = '/home/jvilela/personal_notes/docs/_static_files/images'
rng = np.random.default_rng(42)
DPI = 150
plt.rcParams.update({'font.size': 12})


# ================================================================
# Figure 1: Geometric interpretation — constraint regions
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

center = np.array([0.9, 0.7])       # unconstrained OLS minimum
A = np.array([[4.0, 1.5],           # curvature matrix (tilted ellipses)
              [1.5, 2.0]])

x = np.linspace(-1.4, 1.4, 400)
y = np.linspace(-1.4, 1.4, 400)
W1, W2 = np.meshgrid(x, y)
diff = np.stack([W1 - center[0], W2 - center[1]], axis=-1)
Z = np.einsum('...i,ij,...j->...', diff, A, diff)

t = 0.7   # constraint budget

configs = [
    ('L2 (Ridge) — $\\|\\mathbf{w}\\|_2^2 \\leq t$', True),
    ('L1 (Lasso) — $\\|\\mathbf{w}\\|_1 \\leq t$',   False),
]

for ax, (label, is_l2) in zip(axes, configs):
    # loss contours
    ax.contour(W1, W2, Z,
               levels=[0.05, 0.15, 0.35, 0.65, 1.05, 1.6],
               colors='steelblue', linewidths=1.2, alpha=0.75)

    # constraint region
    if is_l2:
        ax.add_patch(mpatches.Circle((0, 0), t, color='salmon', alpha=0.35, zorder=2))
        ax.add_patch(mpatches.Circle((0, 0), t, fill=False, edgecolor='salmon', linewidth=2.2, zorder=3))
    else:
        verts = [(-t, 0), (0, t), (t, 0), (0, -t)]
        ax.add_patch(mpatches.Polygon(verts, color='salmon', alpha=0.35, zorder=2))
        ax.add_patch(mpatches.Polygon(verts, fill=False, edgecolor='salmon', linewidth=2.2, zorder=3))

    # constrained optimum via SLSQP
    def obj(w):
        d = w - center
        return float(d @ A @ d)

    if is_l2:
        cons = {'type': 'ineq', 'fun': lambda w: t**2 - w[0]**2 - w[1]**2}
    else:
        cons = {'type': 'ineq', 'fun': lambda w: t - abs(w[0]) - abs(w[1])}

    res = minimize(obj, [0.35, 0.35], constraints=cons, method='SLSQP')
    opt = res.x

    ax.plot(*center, '*', color='steelblue', markersize=14, zorder=6, label='OLS solution')
    ax.plot(*opt,    'o', color='red',        markersize=10, zorder=7, label='Regularized solution')

    # annotate: for L1 the solution is on an axis
    if not is_l2:
        ax.annotate('Corner → $w_j = 0$',
                    xy=opt, xytext=(opt[0] + 0.35, opt[1] + 0.35),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10)

    ax.axhline(0, color='gray', lw=0.7)
    ax.axvline(0, color='gray', lw=0.7)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('$w_1$', fontsize=13)
    ax.set_ylabel('$w_2$', fontsize=13)
    ax.set_title(label, fontsize=12.5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')

plt.suptitle('Constraint-Region View of Regularization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/regularization_constraint_regions.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")


# ================================================================
# Figure 2: Penalty functions and their gradients near zero
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

w = np.linspace(-2, 2, 600)
lam = 1.0

# --- L2 ---
ax = axes[0]
ax.plot(w, lam * w**2,     color='steelblue', lw=2,   label=r'$\lambda w^2$ (penalty)')
ax.plot(w, 2 * lam * w,    color='coral',     lw=2,   ls='--', label=r'$2\lambda w$ (gradient)')
ax.axvline(0, color='gray', lw=0.7)
ax.axhline(0, color='gray', lw=0.7)
ax.annotate('gradient $\\to 0$\nas $w \\to 0$',
            xy=(0.05, 0.1), xytext=(0.55, 0.8),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='->', color='dimgray'), fontsize=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-2.5, 4.5)
ax.set_title('L2 Penalty (Ridge)', fontsize=12.5)
ax.set_xlabel('$w$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- L1 ---
ax = axes[1]
eps = 1e-9
mask = np.abs(w) > eps
ax.plot(w, lam * np.abs(w),                       color='steelblue', lw=2,   label=r'$\lambda|w|$ (penalty)')
ax.plot(w[mask], lam * np.sign(w[mask]),           color='coral',     lw=2,   ls='--', label=r'$\lambda\,\mathrm{sign}(w)$ (subgradient)')
# vertical segment at w=0 (the subdifferential [-λ, λ])
ax.plot([0, 0], [-lam, lam],                       color='coral',     lw=2.5, ls='--')
ax.plot(0, 0, 'o', color='coral', markersize=7, zorder=5)
ax.axvline(0, color='gray', lw=0.7)
ax.axhline(0, color='gray', lw=0.7)
ax.annotate('subgradient stays $\\pm\\lambda$\neven as $w \\to 0$',
            xy=(0.05, lam), xytext=(0.45, 1.6),
            xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle='->', color='dimgray'), fontsize=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-2.5, 4.5)
ax.set_title('L1 Penalty (Lasso)', fontsize=12.5)
ax.set_xlabel('$w$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('Penalty Functions and Their Gradients', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/regularization_penalty_gradient.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")


# ================================================================
# Figure 3: Coefficient regularization paths — Lasso vs Ridge
# ================================================================
n_samples, n_features = 120, 10
X, y, true_coef = make_regression(
    n_samples=n_samples, n_features=n_features,
    n_informative=4, noise=20, coef=True, random_state=42)
sc = StandardScaler()
X = sc.fit_transform(X)
y = (y - y.mean()) / y.std()

alphas = np.logspace(-3, 2, 140)
coefs_lasso, coefs_ridge = [], []
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=20000, fit_intercept=False)
    ridge = Ridge(alpha=a,               fit_intercept=False)
    lasso.fit(X, y);  coefs_lasso.append(lasso.coef_)
    ridge.fit(X, y);  coefs_ridge.append(ridge.coef_)

coefs_lasso = np.array(coefs_lasso)
coefs_ridge = np.array(coefs_ridge)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
palette = plt.cm.tab10(np.linspace(0, 1, n_features))

for ax, coefs, title, note in zip(
        axes,
        [coefs_lasso, coefs_ridge],
        ['L1 (Lasso) — coefficient paths',
         'L2 (Ridge) — coefficient paths'],
        ['Lasso drives coefficients\nexactly to zero',
         'Ridge shrinks coefficients\nbut never reaches zero']):

    for i in range(n_features):
        ax.plot(alphas, coefs[:, i], lw=1.6, alpha=0.85, color=palette[i])

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.axhline(0, color='black', lw=0.9, ls='--')
    ax.set_xlabel(r'$\alpha$ (stronger $\rightarrow$)', fontsize=12)
    ax.set_ylabel('Coefficient value', fontsize=12)
    ax.set_title(title, fontsize=12.5)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.04, note,
            ha='right', va='bottom', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Regularization Paths: Lasso vs Ridge', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/regularization_paths.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")


# ================================================================
# Figure 4: Elastic Net — interpolation between L1 and L2
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

l1_ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
colors_en  = ['#2166ac', '#74add1', '#f46d43', '#d73027', '#a50026']
labels_en  = {0.0: 'Ridge  ($\\rho=0$)',
              0.2: '$\\rho=0.2$',
              0.5: 'Elastic Net ($\\rho=0.5$)',
              0.8: '$\\rho=0.8$',
              1.0: 'Lasso  ($\\rho=1$)'}

# pick an informative and a near-zero feature
info_feat  = int(np.argmax(np.abs(true_coef)))
noise_feat = int(np.argmin(np.abs(true_coef)))

for ax, feat, ftitle in zip(
        axes,
        [info_feat,  noise_feat],
        ['Most informative feature', 'Near-zero (noise) feature']):

    for l1r, col in zip(l1_ratios, colors_en):
        coefs_en = []
        for a in alphas:
            if l1r == 0.0:
                m = Ridge(alpha=a, fit_intercept=False)
            elif l1r == 1.0:
                m = Lasso(alpha=a, max_iter=20000, fit_intercept=False)
            else:
                m = ElasticNet(alpha=a, l1_ratio=l1r, max_iter=20000, fit_intercept=False)
            m.fit(X, y)
            coefs_en.append(m.coef_[feat])
        ax.plot(alphas, coefs_en, color=col, lw=2.0,
                label=labels_en[l1r], alpha=0.9)

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.axhline(0, color='black', lw=0.9, ls='--')
    ax.set_xlabel(r'$\alpha$ (stronger $\rightarrow$)', fontsize=12)
    ax.set_ylabel('Coefficient value', fontsize=12)
    ax.set_title(ftitle, fontsize=12.5)
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.suptitle('Elastic Net: Interpolating Between L1 and L2', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/regularization_elastic_net.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

print("\nAll figures generated successfully.")

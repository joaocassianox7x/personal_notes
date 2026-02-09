"""
Toy Transformer – training + figure generation.

Trains a 1-block, 4-head decoder-only transformer on a circular-shift task
and produces publication-quality figures for the documentation.

Outputs (under docs/_static_files/images/):
    transformer_loss_curve.png
    transformer_positional_encoding.png
    transformer_attention_matrix.png
    transformer_attention_heads.png
    transformer_multihead_attention.png
    transformer_block_diagram.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Transformer components  (same as in the .rst, self-contained for plotting)
# ══════════════════════════════════════════════════════════════════════════════

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = 10000.0 ** (2 * (np.arange(d_model)[None, :] // 2) / d_model)
    pe[:, 0::2] = np.sin(pos / div[:, 0::2])
    pe[:, 1::2] = np.cos(pos / div[:, 1::2])
    return pe

def cross_entropy_loss(logits, targets):
    probs = softmax(logits)
    n = logits.shape[0]
    log_probs = np.log(probs[np.arange(n), targets] + 1e-9)
    loss = -log_probs.mean()
    grad = probs.copy()
    grad[np.arange(n), targets] -= 1
    grad /= n
    return loss, grad

# ── Attention ────────────────────────────────────────────────────────────────

def attention_forward(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn = softmax(scores)
    out = attn @ V
    cache = (Q, K, V, attn, scores, mask)
    return out, cache

def attention_backward(d_out, cache):
    Q, K, V, attn, scores, mask = cache
    d_k = Q.shape[-1]
    d_attn = d_out @ V.T
    d_V = attn.T @ d_out
    d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
    d_scores /= np.sqrt(d_k)
    d_Q = d_scores @ K
    d_K = d_scores.T @ Q
    return d_Q, d_K, d_V

# ── Multi-Head Attention ─────────────────────────────────────────────────────

def multihead_attention_forward(X, params, n_heads, mask=None):
    Wq, Wk, Wv, Wo = params['Wq'], params['Wk'], params['Wv'], params['Wo']
    n, d_model = X.shape
    d_k = d_model // n_heads
    Q_all, K_all, V_all = X @ Wq, X @ Wk, X @ Wv
    Q_h = Q_all.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    K_h = K_all.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    V_h = V_all.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    head_outs, head_caches = [], []
    for i in range(n_heads):
        o, c = attention_forward(Q_h[i], K_h[i], V_h[i], mask)
        head_outs.append(o); head_caches.append(c)
    concat = np.concatenate(head_outs, axis=-1)
    output = concat @ Wo
    cache = (X, Q_all, K_all, V_all, concat, head_caches, n_heads, d_k)
    return output, cache

def multihead_attention_backward(d_out, cache, params):
    Wo = params['Wo']
    X, Q_all, K_all, V_all, concat, head_caches, n_heads, d_k = cache
    Wq, Wk, Wv = params['Wq'], params['Wk'], params['Wv']
    n, d_model = X.shape
    d_concat = d_out @ Wo.T
    d_Wo = concat.T @ d_out
    d_heads = np.split(d_concat, n_heads, axis=-1)
    d_Q_all, d_K_all, d_V_all = np.zeros_like(Q_all), np.zeros_like(K_all), np.zeros_like(V_all)
    for i in range(n_heads):
        dq, dk, dv = attention_backward(d_heads[i], head_caches[i])
        d_Q_all[:, i*d_k:(i+1)*d_k] = dq
        d_K_all[:, i*d_k:(i+1)*d_k] = dk
        d_V_all[:, i*d_k:(i+1)*d_k] = dv
    d_Wq, d_Wk, d_Wv = X.T @ d_Q_all, X.T @ d_K_all, X.T @ d_V_all
    d_X = d_Q_all @ Wq.T + d_K_all @ Wk.T + d_V_all @ Wv.T
    return d_X, {'Wq': d_Wq, 'Wk': d_Wk, 'Wv': d_Wv, 'Wo': d_Wo}

# ── FFN ──────────────────────────────────────────────────────────────────────

def ffn_forward(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = X @ W1 + b1; a1 = relu(z1); z2 = a1 @ W2 + b2
    return z2, (X, z1, a1)

def ffn_backward(d_out, cache, params):
    X, z1, a1 = cache
    W1, W2 = params['W1'], params['W2']
    d_a1 = d_out @ W2.T; d_z1 = d_a1 * relu_grad(z1)
    return d_z1 @ W1.T, {
        'W1': X.T @ d_z1, 'b1': d_z1.sum(0),
        'W2': a1.T @ d_out, 'b2': d_out.sum(0),
    }

# ── Layer Norm ───────────────────────────────────────────────────────────────

def ln_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta, (x_hat, gamma, np.sqrt(var + eps))

def ln_backward(d_out, cache):
    x_hat, gamma, std = cache; d = x_hat.shape[-1]
    dx_hat = d_out * gamma
    return (1.0/d)/std*(d*dx_hat - dx_hat.sum(-1,keepdims=True)
        - x_hat*(dx_hat*x_hat).sum(-1,keepdims=True)), \
        (d_out*x_hat).sum(0), d_out.sum(0)

# ── Transformer Block ───────────────────────────────────────────────────────

def block_forward(X, ap, fp, l1, l2, nh, mask=None):
    ao, ac = multihead_attention_forward(X, ap, nh, mask)
    r1 = X + ao; ln1o, ln1c = ln_forward(r1, l1['gamma'], l1['beta'])
    fo, fc = ffn_forward(ln1o, fp)
    r2 = ln1o + fo; ln2o, ln2c = ln_forward(r2, l2['gamma'], l2['beta'])
    return ln2o, (X, ac, ln1c, ln1o, fc, ln2c)

def block_backward(d, cache, ap, fp, l1, l2):
    X, ac, ln1c, ln1o, fc, ln2c = cache
    dr2, dl2g, dl2b = ln_backward(d, ln2c)
    dl1o = dr2.copy(); dfo = dr2.copy()
    dfi, fg = ffn_backward(dfo, fc, fp); dl1o += dfi
    dr1, dl1g, dl1b = ln_backward(dl1o, ln1c)
    dX = dr1.copy(); dao = dr1.copy()
    dai, ag = multihead_attention_backward(dao, ac, ap); dX += dai
    return dX, ag, fg, {'gamma':dl1g,'beta':dl1b}, {'gamma':dl2g,'beta':dl2b}

# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class ToyTransformer:
    def __init__(self, V, dm, nh, dff, ml, seed=42):
        rng = np.random.default_rng(seed); s = np.sqrt(2.0/dm)
        self.dm, self.nh, self.V, self.ml = dm, nh, V, ml
        self.We = rng.normal(scale=0.02, size=(V, dm))
        self.PE = sinusoidal_pe(ml, dm)
        self.ap = {k: rng.normal(scale=s, size=(dm, dm)) for k in ('Wq','Wk','Wv','Wo')}
        self.fp = {'W1': rng.normal(scale=s, size=(dm, dff)), 'b1': np.zeros(dff),
                   'W2': rng.normal(scale=s, size=(dff, dm)), 'b2': np.zeros(dm)}
        self.l1 = {'gamma': np.ones(dm), 'beta': np.zeros(dm)}
        self.l2 = {'gamma': np.ones(dm), 'beta': np.zeros(dm)}
        self.Wo = self.We

    def forward(self, tok, causal=False):
        n = len(tok)
        X = self.We[tok]*np.sqrt(self.dm) + self.PE[:n]
        mask = np.triu(np.full((n,n), -1e9), k=1) if causal else None
        H, bc = block_forward(X, self.ap, self.fp, self.l1, self.l2, self.nh, mask)
        logits = H @ self.Wo.T
        self._c = (tok, X, bc, H); return logits

    def backward(self, dl):
        tok, X, bc, H = self._c; n = len(tok)
        dH = dl @ self.Wo; dWo = dl.T @ H
        dX, ag, fg, l1g, l2g = block_backward(dH, bc, self.ap, self.fp, self.l1, self.l2)
        dWe = np.zeros_like(self.We); sc = np.sqrt(self.dm)
        for i, t in enumerate(tok): dWe[t] += dX[i]*sc
        dWe += dWo
        self._g = {'We':dWe,'ap':ag,'fp':fg,'l1':l1g,'l2':l2g}

    def _init_adam(self):
        """Initialize Adam optimizer state (first and second moments)."""
        self._adam_t = 0
        self._m = {}; self._v = {}
        for name, param in self._all_params():
            self._m[name] = np.zeros_like(param)
            self._v[name] = np.zeros_like(param)

    def _all_params(self):
        """Yield (name, array) for all trainable parameters."""
        yield ('We', self.We)
        for k in ('Wq','Wk','Wv','Wo'):
            yield (f'ap_{k}', self.ap[k])
        for k in ('W1','b1','W2','b2'):
            yield (f'fp_{k}', self.fp[k])
        for k in ('gamma','beta'):
            yield (f'l1_{k}', self.l1[k])
            yield (f'l2_{k}', self.l2[k])

    def _get_grad(self, name):
        g = self._g
        if name == 'We': return g['We']
        if name.startswith('ap_'): return g['ap'][name[3:]]
        if name.startswith('fp_'): return g['fp'][name[3:]]
        if name.startswith('l1_'): return g['l1'][name[3:]]
        if name.startswith('l2_'): return g['l2'][name[3:]]

    def step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam update."""
        if not hasattr(self, '_adam_t'):
            self._init_adam()
        self._adam_t += 1
        t = self._adam_t
        for name, param in self._all_params():
            grad = self._get_grad(name)
            self._m[name] = beta1 * self._m[name] + (1-beta1) * grad
            self._v[name] = beta2 * self._v[name] + (1-beta2) * grad**2
            m_hat = self._m[name] / (1 - beta1**t)
            v_hat = self._v[name] / (1 - beta2**t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self.Wo = self.We  # keep tied

    def get_attention_weights(self, tok, causal=False):
        """Run forward and return per-head attention weight matrices."""
        n = len(tok)
        X = self.We[tok]*np.sqrt(self.dm) + self.PE[:n]
        mask = np.triu(np.full((n,n), -1e9), k=1) if causal else None
        d_k = self.dm // self.nh
        Q = X @ self.ap['Wq']; K = X @ self.ap['Wk']
        Qh = Q.reshape(n, self.nh, d_k).transpose(1,0,2)
        Kh = K.reshape(n, self.nh, d_k).transpose(1,0,2)
        weights = []
        for i in range(self.nh):
            s = Qh[i] @ Kh[i].T / np.sqrt(d_k)
            if mask is not None:
                s = s + mask
            weights.append(softmax(s))
        return weights

# ══════════════════════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════════════════════

print("Training toy transformer on sequence-reversal task …")
# Task: given [a, b, c], predict the reversed sequence [c, b, a].
# We use short sequences (length 4) from a small vocab to keep it tractable.
V, sl, dm, nh, dff = 8, 4, 64, 4, 128
model = ToyTransformer(V, dm, nh, dff, sl+1)
rng = np.random.default_rng(42)

# Pre-generate a fixed training set (memorisation is fine for a toy demo)
n_train = 50
train_data = [rng.integers(0, V, size=sl) for _ in range(n_train)]

losses = []
n_epochs = 4000
for ep in range(n_epochs):
    seq = train_data[ep % n_train]
    tgt = seq[::-1].copy()  # reverse
    logits = model.forward(seq)
    loss, dl = cross_entropy_loss(logits, tgt)
    model.backward(dl); model.step(lr=1e-3)
    losses.append(loss)
    if ep % 200 == 0:
        acc = (logits.argmax(-1) == tgt).mean()
        print(f"  epoch {ep:4d}  loss={loss:.4f}  acc={acc:.2f}")

# Final evaluation on a few examples
print("\n--- Evaluation ---")
correct = 0; total = 0
for seq in train_data[:20]:
    tgt = seq[::-1].copy()
    preds = model.forward(seq).argmax(-1)
    match = np.array_equal(preds, tgt)
    correct += int(match); total += 1
    if total <= 5:
        print(f"  input={seq}  target={tgt}  predicted={preds}  {'✓' if match else '✗'}")
print(f"Accuracy on first 20 training seqs: {correct}/{total}")
print(f"Final loss: {losses[-1]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Training loss curve
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color="#2563eb", linewidth=1.2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title("Toy Transformer Training Loss (Circular-Shift Task)", fontsize=13)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_loss_curve.png"), dpi=150)
plt.close(fig)
print("Saved transformer_loss_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Positional Encoding heatmap
# ══════════════════════════════════════════════════════════════════════════════

pe = sinusoidal_pe(64, 64)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pe, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel("Embedding Dimension $j$", fontsize=12)
ax.set_ylabel("Position $p$", fontsize=12)
ax.set_title("Sinusoidal Positional Encoding", fontsize=13)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_positional_encoding.png"), dpi=150)
plt.close(fig)
print("Saved transformer_positional_encoding.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Attention weight matrix (single head example)
# ══════════════════════════════════════════════════════════════════════════════

example_seq = train_data[0]
weights = model.get_attention_weights(example_seq)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(weights[0], cmap='viridis', vmin=0, vmax=1)
labels = [str(t) for t in example_seq]
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Key (token)", fontsize=11)
ax.set_ylabel("Query (token)", fontsize=11)
ax.set_title("Attention Weights – Head 0", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_attention_matrix.png"), dpi=150)
plt.close(fig)
print("Saved transformer_attention_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – All attention heads
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, nh, figsize=(4*nh, 4))
for i, ax in enumerate(axes):
    im = ax.imshow(weights[i], cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"Head {i}", fontsize=11)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    if i == 0:
        ax.set_ylabel("Query", fontsize=10)
    ax.set_xlabel("Key", fontsize=10)
fig.suptitle("Attention Weight Matrices per Head (after training)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_attention_heads.png"), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved transformer_attention_heads.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Multi-head attention schematic
# ══════════════════════════════════════════════════════════════════════════════

def _rounded_box(ax, xy, w, h, label, color, fontsize=9):
    box = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(box)
    ax.text(xy[0]+w/2, xy[1]+h/2, label, ha='center', va='center',
            fontsize=fontsize, weight='bold')

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title("Multi-Head Self-Attention", fontsize=14, pad=15)

# Input
_rounded_box(ax, (3.5, 0), 3, 0.6, "Input X  (n × d_model)", "#e0e7ff", 10)

# Linear projections
for i, (lbl, x_pos) in enumerate([("W_Q", 1), ("W_K", 4), ("W_V", 7)]):
    _rounded_box(ax, (x_pos, 1.2), 2, 0.5, f"Linear {lbl}", "#bfdbfe")
    ax.annotate("", xy=(x_pos+1, 1.2), xytext=(5, 0.6),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))

# Heads
colors_h = ["#fde68a", "#fed7aa", "#d9f99d", "#c7d2fe"]
for i in range(4):
    xp = 0.5 + i*2.5
    _rounded_box(ax, (xp, 2.2), 2, 0.5, f"Head {i}", colors_h[i])
    ax.annotate("", xy=(xp+1, 2.2), xytext=(xp+1, 1.7),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))

# Concat
_rounded_box(ax, (2.5, 3.2), 5, 0.5, "Concat  (n × d_model)", "#ddd6fe")
for i in range(4):
    xp = 0.5 + i*2.5
    ax.annotate("", xy=(5, 3.2), xytext=(xp+1, 2.7),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# Output projection
_rounded_box(ax, (3, 4.2), 4, 0.5, "Linear W_O", "#bfdbfe")
ax.annotate("", xy=(5, 4.2), xytext=(5, 3.7),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1))

# Output
_rounded_box(ax, (3, 5.2), 4, 0.5, "Output  (n × d_model)", "#bbf7d0", 10)
ax.annotate("", xy=(5, 5.2), xytext=(5, 4.7),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_multihead_attention.png"), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved transformer_multihead_attention.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Transformer block diagram
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 9))
ax.set_xlim(-1, 7); ax.set_ylim(-0.5, 9.5)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title("Transformer Encoder Block", fontsize=14, pad=12)

components = [
    (1.5, 0.2, 3, 0.6, "Input\n(n × d_model)", "#e0e7ff"),
    (1.5, 1.3, 3, 0.7, "Multi-Head\nSelf-Attention", "#fde68a"),
    (1.5, 2.6, 3, 0.5, "Add & LayerNorm", "#d9f99d"),
    (1.5, 3.7, 3, 0.7, "Feed-Forward\nNetwork", "#fed7aa"),
    (1.5, 5.0, 3, 0.5, "Add & LayerNorm", "#d9f99d"),
    (1.5, 6.1, 3, 0.6, "Output\n(n × d_model)", "#bbf7d0"),
]

for (x, y, w, h, lbl, col) in components:
    _rounded_box(ax, (x, y), w, h, lbl, col, 9)

# Vertical arrows
arrows = [(3, 0.8, 3, 1.3), (3, 2.0, 3, 2.6), (3, 3.1, 3, 3.7),
          (3, 4.4, 3, 5.0), (3, 5.5, 3, 6.1)]
for (x1, y1, x2, y2) in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.3))

# Residual connections (curved arrows on the right)
for (y_start, y_end, label) in [(0.5, 2.8, "residual"), (3.2, 5.2, "residual")]:
    ax.annotate("", xy=(5.2, y_end), xytext=(5.2, y_start),
                arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))
    mid_y = (y_start + y_end) / 2
    ax.text(5.9, mid_y, label, fontsize=8, color="#6366f1", rotation=90,
            va='center', ha='center', fontstyle='italic')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transformer_block_diagram.png"), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved transformer_block_diagram.png")

print("\n✓ All figures generated successfully.")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(
    X, Y, y, test_size=0.30, stratify=y, random_state=42)

print(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape)
# ---------- Task 3: forward pass ----------
def init_params(W, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.01, (64, W)), "b1": np.zeros(W),
        "W2": rng.normal(0, 0.01, (W, W)),  "b2": np.zeros(W),
        "W3": rng.normal(0, 0.01, (W, 10)), "b3": np.zeros(10),
    }

def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(0, z2)
    z3 = a2 @ p["W3"] + p["b3"]
    z3 = z3 - z3.max(axis=1, keepdims=True)   # numerical stability
    e = np.exp(z3)
    probs = e / e.sum(axis=1, keepdims=True)
    return probs, (X, z1, a1, z2, a2)

# ---------- Task 4: cross-entropy loss ----------
def loss_fn(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))

# ---------- checks ----------
for W in [16, 64]:
    p = init_params(W)
    probs, cache = forward(Xtr[:32], p)
    print(f"W={W}")
    print("  output shape:", probs.shape)
    print("  rows sum to 1:", np.allclose(probs.sum(axis=1), 1, atol=1e-6))
    print("  untrained loss:", loss_fn(probs, Ytr[:32]))
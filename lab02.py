import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(
    X, Y, y, test_size=0.30, stratify=y, random_state=42)
print(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape)


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


def loss_fn(probs, Y):
    return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))


for W in [16, 64]:
    p = init_params(W)
    probs, cache = forward(Xtr[:32], p)
    print(f"W={W}")
    print("  output shape:", probs.shape)
    print("  rows sum to 1:", np.allclose(probs.sum(axis=1), 1, atol=1e-6))
    print("  untrained loss:", loss_fn(probs, Ytr[:32]))


def backward(probs, cache, Y, p):
    X, z1, a1, z2, a2 = cache
    n = X.shape[0]
    dz3 = (probs - Y) / n                 # softmax + cross-entropy
    g = {}
    g["W3"] = a2.T @ dz3
    g["b3"] = dz3.sum(axis=0)
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (z2 > 0)                  # ReLU derivative
    g["W2"] = a1.T @ dz2
    g["b2"] = dz2.sum(axis=0)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(axis=0)
    return g

def grad_check(p, X, Y, h=1e-5, n_checks=10, seed=1):
    rng = np.random.default_rng(seed)
    probs, cache = forward(X, p)
    grads = backward(probs, cache, Y, p)
    names = ["W1", "W2", "W3"]
    passed = 0
    print(f"{'weight':<14}{'analytic':>14}{'numerical':>14}{'rel. error':>14}")
    for _ in range(n_checks):
        while True:
            name = names[rng.integers(len(names))]
            idx = tuple(int(rng.integers(s)) for s in p[name].shape)
            if abs(grads[name][idx]) > 1e-12:   # skip dead / zero-gradient weights
                break
        orig = p[name][idx]
        p[name][idx] = orig + h
        lp = loss_fn(forward(X, p)[0], Y)
        p[name][idx] = orig - h
        lm = loss_fn(forward(X, p)[0], Y)
        p[name][idx] = orig
        num = (lp - lm) / (2 * h)
        ana = grads[name][idx]
        rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-12)
        passed += rel < 1e-6
        print(f"{name}{str(idx):<10}{ana:>14.4e}{num:>14.4e}{rel:>14.2e}")
    print(f"PASSED {passed}/{n_checks} (need at least 9)")

print("\n--- gradient check, lab init N(0, 0.01) ---")
grad_check(init_params(16), Xtr[:20], Ytr[:20])

print("\n--- gradient check, larger weights (x10) ---")
p_big = {k: v * 10 for k, v in init_params(16).items()}
grad_check(p_big, Xtr[:20], Ytr[:20])
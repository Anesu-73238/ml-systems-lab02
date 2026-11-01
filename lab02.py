import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(
    X, Y, y, test_size=0.30, stratify=y, random_state=42)

print(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape)
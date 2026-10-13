import numpy as np

# Matrice de covariance et valeurs propres (sections 1)
X = np.array([[1, 2], [2, 2], [3, 5]])        # 3 individus, 2 variables
Xc = X - X.mean(axis=0)                        # tableau centré
S = Xc.T @ Xc / len(X)                         # covariance empirique
print(S.round(3))                              # [[0.667 1.] [1. 2.]]
lam, U = np.linalg.eigh(S)                     # valeurs propres croissantes
print(lam.round(4))                            # [0.1315 2.5352]
print(U[:, -1].round(3))                       # premier axe de l'ACP

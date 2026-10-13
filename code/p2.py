import numpy as np

# Droite des moindres carrés (section 2)
x = np.array([1, 2, 3]); y = np.array([2, 2, 5])
b = np.mean((x - x.mean()) * (y - y.mean())) / np.var(x)   # s_xy / s_x²
a = y.mean() - b * x.mean()
print(a, b)                                    # 0.0 1.5
print(np.polyfit(x, y, deg=1).round(4))        # [1.5 0.] : même droite

# Descente de gradient sur L(θ) = (θ - 3)²
theta, eta = 0.0, 0.25
for k in range(6):
    theta = theta - eta * 2 * (theta - 3)
    print(round(theta, 4), end=" ")            # 1.5 2.25 2.625 2.8125 ...

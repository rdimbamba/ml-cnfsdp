import numpy as np
rng = np.random.default_rng(2)

# Loi des grands nombres et TCL (section 3)
p, n, R = 0.3, 400, 20_000                     # R sondages de n personnes
props = rng.binomial(n, p, size=R) / n         # R proportions observées
print(props.mean().round(4), props.var().round(6), p * (1 - p) / n)
ecart = 1.96 * np.sqrt(p * (1 - p) / n)
print(np.mean(np.abs(props - p) <= ecart).round(3))   # proche de 0.95

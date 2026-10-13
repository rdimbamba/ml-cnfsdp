import numpy as np, pandas as pd
rng = np.random.default_rng(3)

# Moyennes conditionnelles par groupe (section 4)
n = 10_000
milieu = rng.choice(["urbain", "rural"], size=n, p=[0.4, 0.6])
enfants = np.where(milieu == "urbain", rng.poisson(2.5, n), rng.poisson(4.5, n))
df = pd.DataFrame({"milieu": milieu, "enfants": enfants})

print(df.groupby("milieu")["enfants"].mean().round(2))   # ≈ 4.5 et 2.5
part = df["milieu"].value_counts(normalize=True)
moy = df.groupby("milieu")["enfants"].mean()
print(round((part * moy).sum(), 4), round(df["enfants"].mean(), 4))  # espérance totale

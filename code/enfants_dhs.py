import numpy as np, pandas as pd

ETIQUETTES = {
    "v024": {1: "région A", 2: "région B", 3: "région C", 4: "région D"},
    "v025": {1: "urbain", 2: "rural"},
    "v106": {0: "aucune", 1: "primaire", 2: "secondaire", 3: "supérieur"},
    "v190": {1: "le plus pauvre", 2: "pauvre", 3: "moyen", 4: "riche", 5: "le plus riche"},
    "b4": {1: "masculin", 2: "féminin"},
    "m15": {11: "domicile", 21: "hôpital public", 31: "clinique privée", 96: "autre"},
    "v467d": {1: "gros problème", 2: "pas un gros problème"},
    "m14": {98: "ne sait pas"},
}

def simuler_enfants(n=3000, graine=0):
    """Enfants de 12 à 23 mois, fictifs, avec les noms et codes de variables du fichier KR des DHS."""
    rng = np.random.default_rng(graine)
    urbain = rng.random(n) < 0.45
    richesse = rng.standard_normal(n) + 1.1 * urbain
    v190 = np.clip(np.digitize(richesse, [-0.9, -0.2, 0.5, 1.2]) + 1, 1, 5)
    v106 = np.clip(np.round(0.9 + 0.7 * richesse + 0.5 * rng.standard_normal(n)), 0, 3).astype(int)
    v024 = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.25, 0.25, 0.15])
    v467d = np.where(rng.random(n) < np.where(urbain, 0.15, 0.55), 1, 2)
    m14 = np.clip(rng.poisson(np.where(urbain, 5.5, 3.5) + 0.6 * (v106 - 1)), 0, 15)
    m15 = np.where(rng.random(n) < 1 / (1 + np.exp(-(0.3 + 1.2 * urbain + 0.5 * v106 - 0.8))),
                   rng.choice([21, 31], n, p=[0.8, 0.2]), rng.choice([11, 96], n, p=[0.9, 0.1]))
    bord = 1 + rng.poisson(np.where(urbain, 1.5, 2.6))
    b4 = rng.choice([1, 2], n)
    hw1 = rng.integers(12, 24, n)
    t = (-0.9 + 0.45 * v106 + 0.35 * (v190 - 3) + 0.35 * urbain - 0.7 * (v467d == 1)
         + 0.12 * np.minimum(m14, 8) + 0.6 * np.isin(m15, [21, 31]) - 0.12 * (bord - 1)
         + np.select([v024 == 4, v024 == 3], [-0.6, 0.3], 0.0))
    p = 1 / (1 + np.exp(-t))
    nb_doses = np.where(rng.random(n) < p, 8, rng.binomial(7, 0.6, n))
    m14 = np.where(rng.random(n) < 0.06, 98, m14)                    # 6 % de « ne sait pas »
    return pd.DataFrame({"caseid": np.arange(1, n + 1), "v024": v024, "v025": np.where(urbain, 1, 2),
                         "v106": v106, "v190": v190, "b4": b4, "bord": bord, "hw1": hw1, "m14": m14,
                         "m15": m15, "v467d": v467d, "nb_doses": nb_doses,
                         "vacc_complet": (nb_doses == 8).astype(int)})

if __name__ == "__main__":
    df = simuler_enfants()
    print(df.head()); print(df.vacc_complet.mean().round(3), df.groupby("v025").vacc_complet.mean().round(3).to_dict())
    print(df.groupby("v106").vacc_complet.mean().round(3).to_dict(), (df.m14 == 98).mean())

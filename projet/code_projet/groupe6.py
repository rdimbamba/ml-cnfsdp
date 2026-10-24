# Groupe 6 - Retard de croissance (fichier KR, un enfant par ligne)
enfant = ["caseid", "midx", "v005", "b0", "b4", "b5", "bord", "b11", "m18",
          "h11", "hw1", "hw70"]
mere = ["v012", "v106", "v190", "v025", "v024", "v136", "v113", "v116",
        "v438", "v445"]
kr = lire_dhs("KR", enfant + mere)

# Population : enfants vivants mesurés ; hw70 >= 9996 = hors limites ou manquant
df = kr[(kr["b5"] == 1) & (kr["hw70"] < 9996)].copy()
df["poids"] = df["v005"] / 1e6

# Cible : score taille-pour-âge (x 100) inférieur à -2 écarts types
df["retard"] = (df["hw70"] < -200).astype(int)
print(len(df), "enfants mesurés ; retard de croissance :",
      round(taux_pondere(df["retard"], df["poids"]), 1), "%")

# Codes spéciaux et recodages
df["taille_mere"] = manquant(df["v438"], range(9990, 10000)) / 10    # en cm
df["imc_mere"] = manquant(df["v445"], range(9990, 10000)) / 100      # en kg/m2
df["m18"] = manquant(df["m18"], [8, 9])   # taille à la naissance : 1 très gros ... 5
df["premier_ne"] = df["b11"].isna().astype(int)  # pas d'intervalle pour un premier-né
df["jumeau"] = (df["b0"] > 0).astype(int)
df["diarrhee"] = indicatrice(df["h11"], [1, 2], manquants=(8, 9))
df["eau_amelioree"] = indicatrice(df["v113"], [11, 12, 13, 14, 21, 31, 41, 51, 71])
df["wc_ameliore"] = indicatrice(df["v116"], [11, 12, 13, 15, 21, 22, 41])
df["urbain"] = (df["v025"] == 1).astype(int)

# Base du supervisé ; à écarter : hw70, hw71, hw72 (indices), hw2, hw3 (mesures)
X, y, w = base_supervisee(df, "retard",
    numeriques=["hw1", "bord", "b11", "m18", "v012", "v136", "taille_mere", "imc_mere",
                "premier_ne", "jumeau", "diarrhee", "eau_amelioree", "wc_ameliore"],
    categorielles=["b4", "v106", "v190", "v025", "v024"])
resume(X, y, w)

# Base du non supervisé : enfant, mère, ménage ; la cible en illustratif
actives = ["hw1", "bord", "m18", "taille_mere", "imc_mere", "v106", "v190", "urbain",
           "v136", "eau_amelioree", "wc_ameliore"]
Z, illus = base_non_supervisee(df, actives, ["retard", "poids"])
display(X.head(), Z.describe().T.round(2))

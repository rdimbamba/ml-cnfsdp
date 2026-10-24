# Groupe 4 - Abandon entre DTC1 et DTC3 (fichier KR, un enfant par ligne)
vaccins = ["h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"]
enfant = ["caseid", "midx", "v005", "v008", "b3", "b4", "b5", "bord"]
mere = ["v012", "v106", "v190", "v025", "v024", "v130", "v136", "v151",
        "v157", "v158", "v159", "m14", "m15"]
kr = lire_dhs("KR", enfant + mere + vaccins)

# Enfants vivants de 12 à 23 mois
kr["age_mois"] = kr["v008"] - kr["b3"]
base = kr[(kr["b5"] == 1) & kr["age_mois"].between(12, 23)].copy()
base["poids"] = base["v005"] / 1e6
base["dtc1"] = base["h3"].isin([1, 2, 3]).astype(int)
base["dtc3"] = base["h7"].isin([1, 2, 3]).astype(int)
print("12-23 mois :", len(base),
      "; DTC1 :", round(taux_pondere(base["dtc1"], base["poids"]), 1),
      "% ; DTC3 :", round(taux_pondere(base["dtc3"], base["poids"]), 1), "%")

# Population : ceux qui ont reçu le DTC1 ; cible : DTC3 non reçu (abandon)
df = base[base["dtc1"] == 1].copy()
df["abandon"] = 1 - df["dtc3"]
print(len(df), "enfants ayant reçu le DTC1 ; abandon :",
      round(taux_pondere(df["abandon"], df["poids"]), 1), "%")

# Codes spéciaux et recodages
for v in ["v157", "v158", "v159"]:
    df[v] = manquant(df[v], [9])
df["m14"] = manquant(df["m14"], [98, 99])
df["accouch_etab"] = indicatrice(df["m15"], range(20, 50))
df["v130"] = manquant(df["v130"], [99])
df["urbain"] = (df["v025"] == 1).astype(int)

# Base du supervisé ; à écarter : h0 à h10 (dont h3, h5, h7), h1 (carte)
X, y, w = base_supervisee(df, "abandon",
    numeriques=["age_mois", "bord", "v012", "v136", "m14", "v157", "v158", "v159",
                "accouch_etab"],
    categorielles=["b4", "v106", "v190", "v025", "v024", "v130", "v151"])
resume(X, y, w)

# Base du non supervisé ; l'abandon en illustratif
actives = ["v012", "v106", "v190", "urbain", "v136", "bord", "v157", "v158", "v159",
           "m14", "accouch_etab"]
Z, illus = base_non_supervisee(df, actives, ["abandon", "poids"])
display(X.head(), Z.describe().T.round(2))

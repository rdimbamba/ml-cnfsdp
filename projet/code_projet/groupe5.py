# Groupe 5 - Vaccination contre la rougeole (fichier KR, un enfant par ligne)
enfant = ["caseid", "midx", "v005", "v008", "b3", "b4", "b5", "bord", "h9"]
mere = ["v012", "v106", "v190", "v025", "v024", "v130", "v136", "v151",
        "v157", "v158", "v159", "m14", "m15"]
kr = lire_dhs("KR", enfant + mere)

# Population : enfants vivants de 12 à 23 mois
kr["age_mois"] = kr["v008"] - kr["b3"]
df = kr[(kr["b5"] == 1) & kr["age_mois"].between(12, 23)].copy()
df["poids"] = df["v005"] / 1e6

# Cible : rougeole reçue (h9 = 1, 2 ou 3) ; 0 = non, 8 = ne sait pas comptent comme non
df["rougeole"] = df["h9"].isin([1, 2, 3]).astype(int)
print(len(df), "enfants de 12-23 mois ; rougeole :",
      round(taux_pondere(df["rougeole"], df["poids"]), 1), "%")

# Codes spéciaux et recodages
for v in ["v157", "v158", "v159"]:
    df[v] = manquant(df[v], [9])
df["m14"] = manquant(df["m14"], [98, 99])
df["accouch_etab"] = indicatrice(df["m15"], range(20, 50))
df["v130"] = manquant(df["v130"], [99])
df["urbain"] = (df["v025"] == 1).astype(int)

# Base du supervisé ; l'âge compte (vaccin vers 9 mois) ; à écarter : h0 à h10
X, y, w = base_supervisee(df, "rougeole",
    numeriques=["age_mois", "bord", "v012", "v136", "m14", "v157", "v158", "v159",
                "accouch_etab"],
    categorielles=["b4", "v106", "v190", "v025", "v024", "v130", "v151"])
resume(X, y, w)

# Base du non supervisé (ACP) ; la cible en illustratif
actives = ["v012", "v106", "v190", "urbain", "v136", "bord", "v157", "v158", "v159",
           "m14", "accouch_etab"]
Z, illus = base_non_supervisee(df, actives, ["rougeole", "poids"])
display(X.head(), Z.describe().T.round(2))

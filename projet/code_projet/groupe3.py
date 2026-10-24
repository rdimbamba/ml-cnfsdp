# Groupe 3 - Vaccination complète (fichier KR, un enfant né depuis 5 ans par ligne)
vaccins = ["h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"]
enfant = ["caseid", "midx", "v005", "v008", "b3", "b4", "b5", "bord"]
mere = ["v012", "v106", "v190", "v025", "v024", "v130", "v136", "v151",
        "v157", "v158", "v159", "m14", "m15"]
kr = lire_dhs("KR", enfant + mere + vaccins)

# Population : enfants vivants (b5 = 1) de 12 à 23 mois ; âge = v008 - b3 (en mois)
kr["age_mois"] = kr["v008"] - kr["b3"]
df = kr[(kr["b5"] == 1) & kr["age_mois"].between(12, 23)].copy()
df["poids"] = df["v005"] / 1e6           # poids de la mère, porté par l'enfant

# Cible : 8 vaccins reçus (1 = date sur carte, 2 = déclaré, 3 = marqué sur carte)
recu = df[vaccins].isin([1, 2, 3])
df["vacc_complete"] = recu.all(axis=1).astype(int)
print(len(df), "enfants de 12-23 mois ; vaccination complète :",
      round(taux_pondere(df["vacc_complete"], df["poids"]), 1), "%")
print("Couverture par vaccin (%) :",
      {v: round(taux_pondere(recu[v], df["poids"]), 1) for v in vaccins})

# Codes spéciaux et recodages
for v in ["v157", "v158", "v159"]:
    df[v] = manquant(df[v], [9])                  # 0 jamais ... 3 presque chaque jour
df["m14"] = manquant(df["m14"], [98, 99])  # visites prénatales (dernière naissance)
df["accouch_etab"] = indicatrice(df["m15"], range(20, 50))   # né en établissement
df["v130"] = manquant(df["v130"], [99])
df["urbain"] = (df["v025"] == 1).astype(int)

# Base du supervisé ; à écarter : h0 à h10, h1 (carte), et tout autre vaccin
X, y, w = base_supervisee(df, "vacc_complete",
    numeriques=["age_mois", "bord", "v012", "v136", "m14", "v157", "v158", "v159",
                "accouch_etab"],
    categorielles=["b4", "v106", "v190", "v025", "v024", "v130", "v151"])
resume(X, y, w)

# Base du non supervisé : profil de la mère et du ménage ; la cible en illustratif
actives = ["v012", "v106", "v190", "urbain", "v136", "bord", "v157", "v158", "v159",
           "m14", "accouch_etab"]
Z, illus = base_non_supervisee(df, actives, ["vacc_complete", "poids"])
display(X.head(), Z.describe().T.round(2))

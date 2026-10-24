# Groupe 8 - Contraception moderne (fichier IR, une femme de 15 à 49 ans par ligne)
femme = ["caseid", "v005", "v502", "v501", "v313", "v012", "v106", "v190", "v025",
         "v024", "v130", "v136", "v201", "v218", "v157", "v158", "v159", "v714",
         "v511", "v743a", "v701", "v505"]
ir = lire_dhs("IR", femme)

# Population : femmes actuellement en union (v502 = 1 : mariées ou en concubinage)
df = ir[ir["v502"] == 1].copy()
df["poids"] = df["v005"] / 1e6                    # poids de sondage de la femme

# Cible : méthode moderne utilisée actuellement (v313 = 3)
df["moderne"] = (df["v313"] == 3).astype(int)
print(len(df), "femmes en union ; contraception moderne :",
      round(taux_pondere(df["moderne"], df["poids"]), 1), "%")

# Codes spéciaux et recodages
for v in ["v157", "v158", "v159"]:
    df[v] = manquant(df[v], [9])
df["v714"] = manquant(df["v714"], [9])            # travaille actuellement : 0 / 1
df["v701"] = manquant(df["v701"], [8, 9])         # instruction du conjoint : 0 à 3
df["v130"] = manquant(df["v130"], [99])
df["polygamie"] = indicatrice(df["v505"], range(1, 98), manquants=(98, 99))
df["decide_sante"] = indicatrice(df["v743a"], [1, 2, 3], manquants=(9,))
df["urbain"] = (df["v025"] == 1).astype(int)

# Base du supervisé ; à écarter : v312, v361 à v364 (méthode, usage passé, prévu),
# v626a (besoins non satisfaits), v602 et v605 (contiennent « stérilisée »)
X, y, w = base_supervisee(df, "moderne",
    numeriques=["v012", "v136", "v201", "v218", "v511", "v157", "v158", "v159",
                "v714", "polygamie", "decide_sante"],
    categorielles=["v106", "v190", "v025", "v024", "v130", "v501", "v701"])
resume(X, y, w)

# Base du non supervisé : profil des femmes ; la cible en illustratif
actives = ["v012", "v106", "v190", "urbain", "v201", "v218", "v511", "v157", "v158",
           "v159", "v714", "v701", "polygamie", "decide_sante"]
Z, illus = base_non_supervisee(df, actives, ["moderne", "poids"])
display(X.head(), Z.describe().T.round(2))

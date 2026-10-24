# Groupe 1 - Richesse : indice national (fichier HR, un ménage par ligne)
biens = ["hv206", "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv221",
         "hv243a", "hv243b", "hv244", "hv246", "hv247"]
logement = ["hv201", "hv205", "hv213", "hv214", "hv215", "hv216", "hv226"]
menage = ["hhid", "hv005", "hv009", "hv010", "hv011", "hv012", "hv014",
          "hv024", "hv025", "hv219", "hv220", "hv270", "hv271"]
hr = lire_dhs("HR", menage + biens + logement)

# Poids et cible
hr["poids"] = hr["hv005"] / 1e6                   # poids de sondage du ménage
hr["pauvre"] = hr["hv270"].isin([1, 2]).astype(int)   # 2 quintiles les plus bas
print("Ménages pauvres :", round(taux_pondere(hr["pauvre"], hr["poids"]), 1), "%")
print("Population pauvre :",
      round(taux_pondere(hr["pauvre"], hr["poids"] * hr["hv012"]), 1), "%")

# Instruction du chef de ménage (fichier PR : hv101 = 1 pour le chef)
pr = lire_dhs("PR", ["hhid", "hv101", "hv106"])
chef = pr[pr["hv101"] == 1].drop_duplicates("hhid")[["hhid", "hv106"]]
hr = hr.merge(chef.rename(columns={"hv106": "instr_chef"}), on="hhid", how="left")
hr["instr_chef"] = manquant(hr["instr_chef"], [8, 9])

# Codes spéciaux : âge du chef 98-99 (ne sait pas, manquant)
hr["hv220"] = manquant(hr["hv220"], [98, 99])

# Variables actives du non supervisé : 1 = possède / logement de qualité
biens = [b for b in biens if b in hr]
for b in biens:
    hr[b] = manquant(hr[b], [9])                  # 0 = non, 1 = oui, 9 = manquant
hr["eau_amelioree"] = indicatrice(hr["hv201"], [11, 12, 13, 14, 21, 31, 41, 51, 71])
hr["wc_ameliore"] = indicatrice(hr["hv205"], [11, 12, 13, 15, 21, 22, 41])
for v, nom in [("hv213", "sol_fini"), ("hv214", "mur_fini"), ("hv215", "toit_fini")]:
    hr[nom] = indicatrice(hr[v], range(31, 40))   # codes 31 à 39 : matériau fini
hr["cuisine_propre"] = indicatrice(hr["hv226"], [1, 2, 3, 4])  # électricité, gaz
hr["pers_par_piece"] = hr["hv009"] / manquant(hr["hv216"], [0, 99])

# Base du supervisé : caractéristiques démographiques (ni biens ni logement)
X, y, w = base_supervisee(hr, "pauvre",
    numeriques=["hv009", "hv010", "hv011", "hv014", "hv220"],
    categorielles=["hv024", "hv025", "hv219", "instr_chef"])
resume(X, y, w)

# Base du non supervisé : biens et logement ; hv270, hv271 et la cible en illustratif
actives = biens + ["eau_amelioree", "wc_ameliore", "sol_fini", "mur_fini",
                   "toit_fini", "cuisine_propre", "pers_par_piece"]
Z, illus = base_non_supervisee(hr, actives, ["pauvre", "hv270", "hv271", "poids"])
display(X.head(), Z.describe().T.round(2))

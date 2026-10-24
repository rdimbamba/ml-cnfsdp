# Groupe 2 - Richesse : indices urbain et rural (fichier HR, un ménage par ligne)
biens = ["hv206", "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv221",
         "hv243a", "hv243b", "hv244", "hv246", "hv247"]
logement = ["hv201", "hv205", "hv213", "hv214", "hv215", "hv216", "hv226"]
menage = ["hhid", "hv005", "hv009", "hv010", "hv011", "hv012", "hv014",
          "hv024", "hv025", "hv219", "hv220", "hv270", "hv271"]
hr = lire_dhs("HR", menage + biens + logement)

# Poids, cible et milieu (hv025 : 1 = urbain, 2 = rural)
hr["poids"] = hr["hv005"] / 1e6
hr["pauvre"] = hr["hv270"].isin([1, 2]).astype(int)
hr["milieu"] = hr["hv025"].map({1: "urbain", 2: "rural"})
for m, d in hr.groupby("milieu"):
    print(f"{m} : {len(d)} ménages ; pauvres",
          round(taux_pondere(d["pauvre"], d["poids"]), 1), "% des ménages,",
          round(taux_pondere(d["pauvre"], d["poids"] * d["hv012"]), 1),
          "% de la population")

# Instruction du chef de ménage (fichier PR : hv101 = 1 pour le chef)
pr = lire_dhs("PR", ["hhid", "hv101", "hv106"])
chef = pr[pr["hv101"] == 1].drop_duplicates("hhid")[["hhid", "hv106"]]
hr = hr.merge(chef.rename(columns={"hv106": "instr_chef"}), on="hhid", how="left")
hr["instr_chef"] = manquant(hr["instr_chef"], [8, 9])
hr["hv220"] = manquant(hr["hv220"], [98, 99])

# Variables actives (mêmes définitions que le groupe 1)
biens = [b for b in biens if b in hr]
for b in biens:
    hr[b] = manquant(hr[b], [9])
hr["eau_amelioree"] = indicatrice(hr["hv201"], [11, 12, 13, 14, 21, 31, 41, 51, 71])
hr["wc_ameliore"] = indicatrice(hr["hv205"], [11, 12, 13, 15, 21, 22, 41])
for v, nom in [("hv213", "sol_fini"), ("hv214", "mur_fini"), ("hv215", "toit_fini")]:
    hr[nom] = indicatrice(hr[v], range(31, 40))
hr["cuisine_propre"] = indicatrice(hr["hv226"], [1, 2, 3, 4])
hr["pers_par_piece"] = hr["hv009"] / manquant(hr["hv216"], [0, 99])
actives = biens + ["eau_amelioree", "wc_ameliore", "sol_fini", "mur_fini",
                   "toit_fini", "cuisine_propre", "pers_par_piece"]

# Une base par milieu : supervisé (sans hv025) et non supervisé
bases = {}
for m in ["urbain", "rural"]:
    d = hr[hr["milieu"] == m]
    X, y, w = base_supervisee(d, "pauvre",
        numeriques=["hv009", "hv010", "hv011", "hv014", "hv220"],
        categorielles=["hv024", "hv219", "instr_chef"])
    Z, illus = base_non_supervisee(d, actives, ["pauvre", "hv270", "hv271", "poids"])
    bases[m] = {"X": X, "y": y, "w": w, "Z": Z, "illus": illus}
    print(m, end=" : "); resume(X, y, w)
# Modèle unique, pour comparer : tous les ménages, avec le milieu hv025 parmi les X
X_tous, y_tous, w_tous = base_supervisee(hr, "pauvre",
    numeriques=["hv009", "hv010", "hv011", "hv014", "hv220"],
    categorielles=["hv024", "hv025", "hv219", "instr_chef"])
display(bases["urbain"]["X"].head(), bases["rural"]["Z"].describe().T.round(2))

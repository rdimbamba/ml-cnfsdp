"""Construit notebooks/Projets_demarrage.ipynb à partir des codes de projet/code_projet/ :
bloc commun, préparation des 9 groupes, vérification des taux pondérés, volet non supervisé.
Usage : python make_projets_demarrage.py [dossier_des_codes] [dossier_de_sortie]"""
import sys, os
import nbformat as nbf

src = sys.argv[1] if len(sys.argv) > 1 else "../projet/code_projet"
sortie = sys.argv[2] if len(sys.argv) > 2 else "../notebooks"
lire = lambda f: open(os.path.join(src, f), encoding="utf-8").read().strip("\n")

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md(r"""
# Projet de groupe : codes de démarrage des neuf groupes

*Introduction au Machine Learning, CNFSDP 2026-2027, Rois Céti DIMBAMBA*

Ce notebook rassemble les codes de préparation du **guide du projet** (section « Fiches de démarrage ») : pour chaque groupe, il lit le fichier de l'EDS Congo 2011-2012, construit la population et la cible, calcule la **proportion pondérée** de la cible, et prépare les bases prêtes à l'emploi :

- `X`, `y`, `w` : variables explicatives, cible et poids, pour le volet supervisé ;
- `Z`, `illus` : variables actives et variables illustratives (cible, poids), pour le volet non supervisé.

Une cellule de **vérification** suit chaque groupe : elle compare l'effectif et le taux pondéré obtenus aux valeurs annoncées dans le guide. Le tableau final récapitule les neuf groupes.

**Avant de lancer.**
1. Décompresser les quatre archives (`CGHR61DT.zip`, `CGIR61DT.zip`, `CGKR61DT.zip`, `CGPR61DT.zip`) dans un dossier `donnees/`, à côté de ce notebook (sous-dossiers acceptés). Le code trouve seul les fichiers `.DTA`.
2. Lancer la cellule du bloc commun, puis les cellules d'un groupe (chaque groupe est indépendant des autres).

**Rappel.** Les données DHS ne sont jamais déposées sur GitHub ni sur aucun dossier partagé. Ce notebook ne contient aucune donnée : seulement du code.

**Les poids.** `hv005`/$10^6$ pour les ménages (fichier HR), `v005`/$10^6$ pour les femmes (IR) et les enfants (KR, poids de la mère). Pour une part de population à partir du fichier des ménages : `hv005`/$10^6$ × `hv012` (membres du ménage). Proportion pondérée : $\hat p=\sum_i w_iy_i/\sum_i w_i$, en Python `np.average(y, weights=w)`.
""")

md("## Bloc commun")
code(lire("bloc_commun.py"))
code(r"""
# Vérification : effectif et taux pondéré comparés aux valeurs du guide
verifications = []
def verifier(groupe, libelle, obtenu, attendu, n=None, n_attendu=None, tolerance=0.3):
    ok_taux = abs(obtenu - attendu) <= tolerance
    ok_n = (n is None) or (n == n_attendu)
    verifications.append({"groupe": groupe, "indicateur": libelle, "obtenu (%)": round(obtenu, 1),
                          "attendu (%)": attendu, "effectif": n, "effectif attendu": n_attendu,
                          "conforme": "oui" if ok_taux and ok_n else "NON"})
    print(f"{libelle} : {obtenu:.1f} % (attendu {attendu} %)",
          "" if n is None else f"; effectif {n} (attendu {n_attendu})",
          "-> conforme" if ok_taux and ok_n else "-> À VÉRIFIER")
""")

groupes = [
 (1, "Richesse : indice national", "HR",
  "ménages pauvres : `hv270` ∈ {1, 2}", r"""
verifier(1, "ménages pauvres (poids hv005)", taux_pondere(hr["pauvre"], hr["poids"]), 40.8,
         len(hr), 11632)
verifier(1, "population pauvre (poids hv005 x hv012)",
         taux_pondere(hr["pauvre"], hr["poids"] * hr["hv012"]), 39.7)
print("Sans pondération :", round(100 * hr["pauvre"].mean(), 1), "% des ménages")
"""),
 (2, "Richesse : indices urbain et rural", "HR",
  "ménages pauvres (`hv270` ∈ {1, 2}), urbain et rural séparés (`hv025`)", r"""
for m, attendu, n_att in [("urbain", 13.9, 3150), ("rural", 86.5, 8482)]:
    d = hr[hr["milieu"] == m]
    verifier(2, f"population {m} pauvre (poids hv005 x hv012)",
             taux_pondere(d["pauvre"], d["poids"] * d["hv012"]), attendu, len(d), n_att)
"""),
 (3, "Vaccination complète", "KR",
  "les 8 vaccins de base reçus (`h2` à `h9` ∈ {1, 2, 3}), enfants vivants de 12 à 23 mois", r"""
verifier(3, "vaccination complète (poids v005)", taux_pondere(df["vacc_complete"], df["poids"]),
         38.1, len(df), 1884)
print("Sans pondération :", round(100 * df["vacc_complete"].mean(), 1), "%")
"""),
 (4, "Abandon entre DTC1 et DTC3", "KR",
  "DTC1 reçu (`h3` ∈ {1, 2, 3}) mais pas le DTC3 (`h7` ∉ {1, 2, 3}), enfants de 12 à 23 mois", r"""
verifier(4, "DTC1, 12-23 mois (poids v005)", taux_pondere(base["dtc1"], base["poids"]), 74.5,
         len(base), 1884)
verifier(4, "DTC3, 12-23 mois (poids v005)", taux_pondere(base["dtc3"], base["poids"]), 56.3)
verifier(4, "abandon parmi les DTC1 (poids v005)", taux_pondere(df["abandon"], df["poids"]), 25.0,
         len(df), 1290)
"""),
 (5, "Vaccination contre la rougeole", "KR",
  "rougeole reçue (`h9` ∈ {1, 2, 3}), enfants vivants de 12 à 23 mois", r"""
verifier(5, "rougeole (poids v005)", taux_pondere(df["rougeole"], df["poids"]), 74.9, len(df), 1884)
"""),
 (6, "Retard de croissance", "KR",
  "`hw70` < -200, enfants vivants mesurés (`hw70` < 9996)", r"""
verifier(6, "retard de croissance (poids v005)", taux_pondere(df["retard"], df["poids"]), 23.1,
         len(df), 4475)
"""),
 (7, "Insuffisance pondérale", "KR",
  "`hw71` < -200, enfants vivants mesurés (`hw71` < 9996)", r"""
verifier(7, "insuffisance pondérale (poids v005)", taux_pondere(df["insuff_pond"], df["poids"]),
         11.1, len(df), 4475)
"""),
 (8, "Contraception moderne", "IR",
  "méthode moderne (`v313` = 3), femmes en union (`v502` = 1)", r"""
verifier(8, "contraception moderne (poids v005)", taux_pondere(df["moderne"], df["poids"]), 20.0,
         len(df), 6750)
"""),
 (9, "Besoins non satisfaits en contraception", "IR",
  "besoin non satisfait (`v626a` ∈ {1, 2}), femmes en union (`v502` = 1)", r"""
verifier(9, "besoins non satisfaits (poids v005)", taux_pondere(df["besoin_ns"], df["poids"]), 18.4,
         len(df), 6750)
"""),
]
for g, titre, fichier, cible, verif in groupes:
    md(f"## Groupe {g} : {titre}\n\nFichier `{fichier}` ; cible : {cible}. Détails dans la fiche du groupe (guide du projet).")
    code(lire(f"groupe{g}.py"))
    code(verif)

md(r"""
## Récapitulatif des vérifications

Un écart de plus de 0,3 point, ou un effectif différent, signale une erreur dans la population ou dans la cible (filtre oublié, code spécial non traité, mauvais poids) : relire la fiche du groupe et le dictionnaire (`.MAP`).
""")
code(r"""
display(pd.DataFrame(verifications).astype({"effectif": "Int64", "effectif attendu": "Int64"}))
""")

md(r"""
## Volet non supervisé (commun à tous les groupes)

À lancer après les cellules d'un groupe : il utilise les tableaux `Z` (variables actives) et `illus` (cible et poids) de ce groupe. Pour le groupe 2, choisir d'abord un milieu : `Z, illus = bases["urbain"]["Z"], bases["urbain"]["illus"]`. Ajuster `q` (nombre d'axes) d'après l'éboulis et `k` (nombre de classes) d'après le dendrogramme.
""")
code(lire("non_supervise.py"))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
os.makedirs(sortie, exist_ok=True)
nbf.write(nb, os.path.join(sortie, "Projets_demarrage.ipynb"))
print("écrit :", os.path.join(sortie, "Projets_demarrage.ipynb"))

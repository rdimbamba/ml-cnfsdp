import nbformat as nbf
from pathlib import Path
md = nbf.v4.new_markdown_cell; co = nbf.v4.new_code_cell
SIMU = open(Path(__file__).with_name("enfants_dhs.py")).read()
SIMU = SIMU[:SIMU.index('if __name__')].replace("import numpy as np, pandas as pd\n", "").strip()

def build(corrige):
    c = []
    def q(sol, squel): c.append(co(sol if corrige else squel))
    t = "TP5 — Un projet de bout en bout" + (" (corrigé)" if corrige else "")
    c.append(md(f"""# {t}
**CNFSDP — Introduction au Machine Learning** · L3 S5 · 2026-2027 · Rois Céti DIMBAMBA · chapitre 5 · 45 min en salle

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/{'corriges/TP5_corrige' if corrige else 'TP5'}.ipynb)

On fabrique d'abord un **faux fichier « enfants » au format DHS** (Stata et SPSS, codes et étiquettes), puis on le traite exactement comme le vrai fichier de l'EDS Congo que vous utiliserez pour le projet.
Les données sont fictives : aucune ne provient d'une enquête réelle. **Les vraies données DHS ne doivent jamais être déposées sur GitHub ni partagées.**"""))
    c.append(co("""# Lecture des fichiers SPSS : pyreadstat (déjà présent dans l'environnement ml-cnfsdp ; à installer sur Colab)
try:
    import pyreadstat
except ImportError:
    %pip install -q pyreadstat
    import pyreadstat"""))
    c.append(co("""import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import roc_auc_score, recall_score, precision_score
VERT, OR, GRIS = "#126B2A", "#B8860B", "#8A8A8A"   # couleurs du cours"""))
    c.append(md("Fabrication du fichier fictif (à exécuter sans le modifier) :"))
    c.append(co(SIMU + """

os.makedirs("donnees", exist_ok=True)
enfants = simuler_enfants()
enfants.to_stata("donnees/ZZKR_FICTIF.DTA", write_index=False, version=118,
                 value_labels={k: v for k, v in ETIQUETTES.items() if k != "m14"},
                 variable_labels={"v106": "niveau d'instruction de la mère", "v025": "milieu de résidence",
                                  "m14": "nombre de visites prénatales", "vacc_complet": "vaccination complète"})
pyreadstat.write_sav(enfants, "donnees/ZZKR_FICTIF.SAV", variable_value_labels=ETIQUETTES)
print(os.listdir("donnees"))"""))

    c.append(md("""## Question 1 — Lire un fichier DHS
Lire les étiquettes avec `pd.io.stata.StataReader`, puis seulement les colonnes utiles avec `pd.read_stata(..., columns=..., convert_categoricals=False)`. Recoder le code 98 de `m14` (« ne sait pas ») en valeur manquante. Relire le fichier SPSS avec `pd.read_spss` et vérifier qu'il a la même taille. Donner le taux de vaccination complète par milieu (codes 1 = urbain, 2 = rural)."""))
    q("""fichier = "donnees/ZZKR_FICTIF.DTA"
with pd.io.stata.StataReader(fichier) as lecteur:
    etiquettes = lecteur.value_labels()
    libelles = lecteur.variable_labels()
print(libelles["v106"], etiquettes["v106"])
print(etiquettes["m15"])

colonnes = ["v024", "v025", "v106", "v190", "b4", "bord", "hw1", "m14", "m15", "v467d", "nb_doses", "vacc_complet"]
df = pd.read_stata(fichier, columns=colonnes, convert_categoricals=False)
df["m14"] = df["m14"].replace(98, np.nan)
print(df.shape, "| manquants m14 :", df["m14"].isna().sum())
spss = pd.read_spss("donnees/ZZKR_FICTIF.SAV", usecols=colonnes)
print("SPSS :", spss.shape)
print(df.groupby("v025")["vacc_complet"].mean().round(3).rename(index=etiquettes["v025"]))""",
"""fichier = "donnees/ZZKR_FICTIF.DTA"
colonnes = ["v024", "v025", "v106", "v190", "b4", "bord", "hw1", "m14", "m15", "v467d", "nb_doses", "vacc_complet"]
df = pd.read_stata(fichier, columns=colonnes, convert_categoricals=False)
# À compléter : étiquettes (StataReader), recodage de 98 en manquant, lecture SPSS, taux de vaccination par milieu
""")

    c.append(md("""## Question 2 — Pipeline et comparaison de modèles
Variables qualitatives : `v024, v025, b4, m15, v467d` ; numériques : `v106, v190, bord, hw1, m14`. Découper (70 % / 30 %, stratifié, `random_state=0`), construire la préparation (indicatrices ; imputation par la médiane puis standardisation), comparer logistique, forêt et gradient boosting par AUC en validation croisée. Évaluer ensuite le meilleur **une seule fois** sur le test : AUC et exactitude avec son intervalle de confiance à 95 %."""))
    q("""qual = ["v024", "v025", "b4", "m15", "v467d"]; quant = ["v106", "v190", "bord", "hw1", "m14"]
df[quant] = df[quant].astype(float)
X, y = df[qual + quant], df["vacc_complet"]
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
prep = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), qual),
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), quant))
modeles = {"logistique": LogisticRegression(max_iter=1000),
           "forêt": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=0),
           "boosting": GradientBoostingClassifier(random_state=0)}
for nom, m in modeles.items():
    auc = cross_val_score(make_pipeline(prep, m), Xa, ya, cv=5, scoring="roc_auc")
    print(f"{nom:10s} AUC = {auc.mean():.3f} (écart type {auc.std():.3f})")

f = make_pipeline(prep, LogisticRegression(max_iter=1000)).fit(Xa, ya)
p = f.score(Xt, yt); m = len(yt)
print(f"test : AUC = {roc_auc_score(yt, f.predict_proba(Xt)[:, 1]):.3f} | exactitude = {p:.3f} "
      f"± {1.96 * np.sqrt(p * (1 - p) / m):.3f}")""",
"""qual = ["v024", "v025", "b4", "m15", "v467d"]; quant = ["v106", "v190", "bord", "hw1", "m14"]
df[quant] = df[quant].astype(float)
X, y = df[qual + quant], df["vacc_complet"]
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
prep = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), qual),
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), quant))
# À compléter : trois modèles en validation croisée (scoring="roc_auc"), puis f = meilleur pipeline ajusté sur
# (Xa, ya) et évaluation unique sur le test (AUC, exactitude ± 1,96 * sqrt(p(1-p)/m))
""")

    c.append(md("""## Question 3 — Deux fuites d'information
1. Ajouter `nb_doses` aux variables numériques et refaire la validation croisée de la logistique. Pourquoi ce score est-il trompeur ?
2. Sur 200 individus, 2 000 variables de bruit et une cible au hasard : choisir les 20 variables les plus liées à $y$ (`SelectKBest`) **sur toutes les données** puis faire une validation croisée ; recommencer avec la sélection **dans** le pipeline."""))
    q("""prep2 = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), qual),
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), quant + ["nb_doses"]))
X2 = df.loc[Xa.index, qual + quant + ["nb_doses"]]
auc = cross_val_score(make_pipeline(prep2, LogisticRegression(max_iter=1000)), X2, ya, cv=5, scoring="roc_auc")
print("avec nb_doses : AUC =", auc.mean().round(3))

rng = np.random.default_rng(0)
B = rng.standard_normal((200, 2000)); yb = rng.integers(0, 2, 200)
Bs = SelectKBest(f_classif, k=20).fit_transform(B, yb)                       # sélection sur tout : fuite
print("sélection avant la validation croisée :", cross_val_score(LogisticRegression(), Bs, yb, cv=5).mean().round(3))
honnete = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())   # sélection dans chaque bloc
print("sélection dans le pipeline :", cross_val_score(honnete, B, yb, cv=5).mean().round(3))""",
"""# À compléter : 1) logistique avec nb_doses en plus (validation croisée, AUC)
# 2) bruit : B = rng.standard_normal((200, 2000)), yb = rng.integers(0, 2, 200) ; SelectKBest(f_classif, k=20)
#    avant la validation croisée, puis dans un pipeline
rng = np.random.default_rng(0)
""")

    c.append(md("""## Question 4 — Interpréter le modèle
Calculer l'importance par permutation (AUC, 20 répétitions) sur le test et la tracer. Calculer la dépendance partielle de `v106` (instruction de la mère) : probabilité moyenne prédite quand on impose 0, 1, 2 puis 3 à tous les enfants."""))
    q("""r = permutation_importance(f, Xt, yt, scoring="roc_auc", n_repeats=20, random_state=0)
imp = pd.Series(r.importances_mean, index=X.columns).sort_values()
imp.plot.barh(color=VERT, figsize=(5, 3)); plt.xlabel("baisse de l'AUC"); plt.show()

pdp = partial_dependence(f, Xa, ["v106"], kind="average")
print(dict(zip(pdp["grid_values"][0], pdp["average"][0].round(3))))""",
"""# À compléter : permutation_importance(f, Xt, yt, scoring="roc_auc", n_repeats=20, random_state=0), graphique,
# puis partial_dependence(f, Xa, ["v106"], kind="average")
""")

    c.append(md("""## Question 5 — Confidentialité
1. Quelle part des enfants est seule dans sa case quand on croise région, milieu, sexe, âge en mois, rang de naissance et instruction de la mère ?
2. Le nombre d'enfants non vaccinés de la région D doit être publié avec le mécanisme de Laplace. Tirer 5 publications pour $\\varepsilon=1$ puis pour $\\varepsilon=0{,}1$ (`rng.laplace(0, 1/eps)`)."""))
    q("""cles = ["v024", "v025", "b4", "hw1", "bord", "v106"]
taille = df.groupby(cles)["v025"].transform("size")
print(f"enfants uniques : {(taille == 1).mean():.1%} | dans une case d'au plus 3 : {(taille <= 3).mean():.1%}")

N = int(((df["v024"] == 4) & (df["vacc_complet"] == 0)).sum())
rng = np.random.default_rng(1)
print("vrai nombre :", N)
for eps in [1, 0.1]:
    print(f"epsilon = {eps} :", np.round(N + rng.laplace(0, 1 / eps, 5)).astype(int))""",
"""cles = ["v024", "v025", "b4", "hw1", "bord", "v106"]
# À compléter : taille de la case de chaque enfant (groupby(...).transform("size")), parts d'uniques et de cases <= 3
# puis N = nombre de non vaccinés de la région D (v024 == 4) et 5 publications bruitées pour chaque epsilon
rng = np.random.default_rng(1)
""")

    c.append(md("""## Question 6 — Audit d'équité
Le modèle « signale » un enfant si sa probabilité prédite de vaccination complète est inférieure à 0,5. Par milieu, calculer la part d'enfants signalés, le rappel et la précision (positif = non vacciné). Chercher ensuite, pour chaque milieu, le seuil qui donne un rappel d'environ 0,75, et recalculer l'audit.

*Remarque : en pratique, on choisit ces seuils sur l'apprentissage (par validation croisée), pas sur le test.*"""))
    q("""p_non = 1 - f.predict_proba(Xt)[:, 1]          # probabilité prédite d'être non vacciné
non_vacc = (yt == 0).values
def audit(seuils):
    for g, milieu in [(1, "urbain"), (2, "rural")]:
        m = (Xt["v025"] == g).values
        s = p_non[m] > seuils[g]
        print(f"{milieu:7s} seuil {seuils[g]:.2f} | signalés {s.mean():.0%} | rappel {recall_score(non_vacc[m], s):.2f}"
              f" | précision {precision_score(non_vacc[m], s):.2f}")
audit({1: 0.5, 2: 0.5})

seuils = {}
for g in [1, 2]:
    m = (Xt["v025"] == g).values
    seuils[g] = np.quantile(p_non[m][non_vacc[m]], 0.25)    # 75 % des non vaccinés au-dessus du seuil
print()
audit(seuils)""",
"""# p_non = 1 - f.predict_proba(Xt)[:, 1]        # probabilité prédite d'être non vacciné
# non_vacc = (yt == 0).values
# À compléter : fonction audit(seuils) par milieu (signalés, recall_score, precision_score), audit au seuil 0,5,
# puis seuils par milieu donnant un rappel d'environ 0,75 (quantile 0,25 de p_non chez les non vaccinés)
""")
    c.append(md("**Conclusion attendue.** La logistique fait aussi bien que les méthodes d'ensemble sur ces données. Une variable qui contient la réponse ou une sélection faite avant le découpage donne des scores faussement excellents. Le modèle repère mal les enfants urbains non vaccinés : avec un seuil par milieu, on égalise les rappels, au prix de plus de signalements en ville. Ces choix se justifient dans le rapport." if corrige else "**À commenter :** quel critère d'équité vous semble le plus important pour une campagne de vaccination, et pourquoi ?"))
    nb = nbf.v4.new_notebook(); nb["cells"] = c
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb

out = Path(__file__).resolve().parent.parent / "notebooks"
nbf.write(build(False), out / "TP5.ipynb"); nbf.write(build(True), out / "corriges" / "TP5_corrige.ipynb")
print("ok")

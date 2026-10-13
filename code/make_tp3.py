import nbformat as nbf
from pathlib import Path
md = nbf.v4.new_markdown_cell; co = nbf.v4.new_code_cell

def q(c, corrige, sol, squel):
    c.append(co(sol if corrige else squel))

def build(corrige):
    c = []
    t = "TP3 — Classifieurs et méthodes d'ensemble" + (" (corrigé)" if corrige else "")
    c.append(md(f"""# {t}
**CNFSDP — Introduction au Machine Learning** · L3 S5 · 2026-2027 · Rois Céti DIMBAMBA · chapitre 3 · 45 min en salle

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/{'corriges/TP3_corrige' if corrige else 'TP3'}.ipynb)

Partie A : plus proches voisins et régression logistique. Partie B : arbres et indice de Gini. Partie C : bagging, forêt, AdaBoost.
Données : le fil rouge du cours (10 enfants, chiffres fictifs), des « croissants » simulés et `breast_cancer`, fourni avec scikit-learn (aucun téléchargement)."""))
    c.append(co("""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
VERT, OR, GRIS = "#126B2A", "#B8860B", "#8A8A8A"   # couleurs du cours

# Fil rouge : études de la mère (ans), distance au centre de santé (km), vacciné
X0 = np.array([[0, 9], [1, 3], [3, 7], [4, 2], [6, 8], [6, 4], [7, 9], [9, 6], [10, 1], [2, 5]])
y0 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0])"""))

    c.append(md("""## Partie A — Plus proches voisins et régression logistique
### Question 1 — $k$-NN à la main
Pour le nouvel enfant $x=(5,5)$ : calculer les distances au carré aux 10 enfants, trier, puis donner $\\hat\\eta(x)$ et la prédiction pour $k=1,3,5$. Vérifier avec `KNeighborsClassifier`."""))
    q(c, corrige, """x = np.array([5, 5])
d2 = ((X0 - x) ** 2).sum(axis=1)                 # distances au carré
ordre = np.argsort(d2, kind="stable")
print("enfants par distance :", ordre + 1, "| d² :", d2[ordre])
for k in [1, 3, 5]:
    eta = y0[ordre[:k]].mean()
    sk = KNeighborsClassifier(n_neighbors=k).fit(X0, y0).predict_proba([x])[0, 1]
    print(f"k = {k} : eta = {eta:.3f} -> prédiction {int(eta > 0.5)} | scikit-learn : {sk:.3f}")""",
"""x = np.array([5, 5])
# À compléter : distances au carré, tri (np.argsort), proportion de 1 parmi les k premiers,
# puis comparaison avec KNeighborsClassifier(n_neighbors=k).fit(X0, y0).predict_proba([x])
""")
    c.append(md("""### Question 2 — Choisir $k$ par validation croisée
Sur 300 points en croissants (`make_moons(noise=0.3)`), calculer l'erreur de validation croisée à 5 blocs pour $k=1,\\dots,50$, tracer la courbe et donner le meilleur $k$. Mesurer ensuite l'erreur test sur 5 000 nouveaux points."""))
    q(c, corrige, """X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
Xt, yt = make_moons(n_samples=5000, noise=0.3, random_state=1)
ks = range(1, 51)
err = [1 - cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean() for k in ks]
k_opt = ks[int(np.argmin(err))]
plt.figure(figsize=(6, 3)); plt.plot(ks, err, color=VERT, marker="o", ms=3)
plt.xlabel("k"); plt.ylabel("erreur de validation croisée"); plt.show()
f = KNeighborsClassifier(n_neighbors=k_opt).fit(X, y)
print("k choisi :", k_opt, "| erreur test :", round(np.mean(f.predict(Xt) != yt), 4))""",
"""X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
Xt, yt = make_moons(n_samples=5000, noise=0.3, random_state=1)
# À compléter : erreur de CV (1 - cross_val_score(...).mean()) pour k = 1..50, tracé, k optimal, erreur test
""")
    c.append(md("""### Question 3 — Régression logistique sur le fil rouge
Avec la seule variable « études », ajuster une régression logistique sans pénalité (`LogisticRegression(C=1e8)`). Retrouver $\\hat\\theta\\approx(-1{,}46;\\,0{,}43)$, la frontière $-\\hat\\theta_0/\\hat\\theta_1$, la cote $e^{\\hat\\theta_1}$ et la perte logistique moyenne. Vérifier que le gradient $\\frac1n\\sum_i(\\hat\\eta_i-y_i)x_{ij}$ est (presque) nul."""))
    q(c, corrige, """xe = X0[:, [0]]                                  # années d'études seules
lr = LogisticRegression(C=1e8).fit(xe, y0)
t0, t1 = lr.intercept_[0], lr.coef_[0, 0]
eta = lr.predict_proba(xe)[:, 1]
perte = -np.mean(y0 * np.log(eta) + (1 - y0) * np.log(1 - eta))
print(f"theta = ({t0:.2f} ; {t1:.2f}) | frontière : {-t0 / t1:.2f} ans | cote : {np.exp(t1):.2f}")
print(f"perte logistique moyenne : {perte:.3f} (constante 0,6 : {-np.mean(y0*np.log(.6)+(1-y0)*np.log(.4)):.3f})")
print("gradient :", np.mean(eta - y0).round(5), np.mean((eta - y0) * xe[:, 0]).round(5))""",
"""xe = X0[:, [0]]                                  # années d'études seules
# À compléter : LogisticRegression(C=1e8), intercept_ et coef_, frontière, cote, perte logistique, gradient
""")

    c.append(md("""## Partie B — Arbres et indice de Gini
### Question 4 — La meilleure coupure
Écrire une fonction `gini(y)` qui renvoie $2p(1-p)$, puis une fonction qui calcule l'impureté moyenne d'une coupure « $x_j\\le s$ ». Essayer tous les seuils (milieux entre valeurs consécutives) des deux variables du fil rouge et retrouver la coupure du cours (distance $\\le4{,}5$, impureté $0{,}267$). Comparer avec l'arbre de profondeur 2 de scikit-learn."""))
    q(c, corrige, """def gini(y):
    p = y.mean()
    return 2 * p * (1 - p)

def impurete(X, y, j, s):
    g = X[:, j] <= s
    return g.mean() * gini(y[g]) + (1 - g.mean()) * gini(y[~g])

resultats = []
for j, nom in enumerate(["études", "distance"]):
    v = np.unique(X0[:, j])
    for s in (v[:-1] + v[1:]) / 2:
        resultats.append((impurete(X0, y0, j, s), nom, s))
for r in sorted(resultats)[:4]:
    print(f"{r[1]:8s} <= {r[2]:4.1f} : impureté {r[0]:.3f}")

arbre = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X0, y0)
print(export_text(arbre, feature_names=["études", "distance"]))""",
"""def gini(y):
    # À compléter
    pass

def impurete(X, y, j, s):
    # À compléter : impureté moyenne de la coupure X[:, j] <= s
    pass

# À compléter : boucle sur les deux variables et les seuils, tri, puis arbre de profondeur 2 (export_text)
""")

    c.append(md("""## Partie C — Méthodes d'ensemble sur `breast_cancer`
569 tumeurs, 30 mesures ; $y=1$ si la tumeur est bénigne. Découpage 70 % / 30 %, stratifié."""))
    c.append(co("""D = load_breast_cancer()
Xa, Xt, ya, yt = train_test_split(D.data, D.target, test_size=0.3, random_state=0, stratify=D.target)
print(Xa.shape, Xt.shape)"""))
    c.append(md("""### Question 5 — Le bagging écrit en NumPy
Pour $B=100$ : tirer un échantillon bootstrap (indices avec remise), y apprendre un arbre profond, prédire le test ; agréger par vote majoritaire. Comparer l'exactitude test avec un arbre seul et avec `BaggingClassifier`. Tracer l'exactitude du vote en fonction du nombre d'arbres."""))
    q(c, corrige, """rng = np.random.default_rng(0)
B, n = 100, len(ya)
votes = np.zeros(len(yt))
exactitude = []
for b in range(1, B + 1):
    idx = rng.integers(0, n, n)                          # échantillon bootstrap
    arbre_b = DecisionTreeClassifier(random_state=b).fit(Xa[idx], ya[idx])
    votes += arbre_b.predict(Xt)
    exactitude.append(np.mean((votes / b > 0.5) == yt))  # vote majoritaire des b premiers arbres

seul = DecisionTreeClassifier(random_state=0).fit(Xa, ya).score(Xt, yt)
sk = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, random_state=0).fit(Xa, ya).score(Xt, yt)
print(f"arbre seul {seul:.3f} | bagging NumPy {exactitude[-1]:.3f} | BaggingClassifier {sk:.3f}")
plt.figure(figsize=(6, 3)); plt.plot(range(1, B + 1), exactitude, color=VERT)
plt.axhline(seul, color=GRIS, ls="--"); plt.xlabel("nombre d'arbres"); plt.ylabel("exactitude test"); plt.show()""",
"""rng = np.random.default_rng(0)
B, n = 100, len(ya)
votes = np.zeros(len(yt))
exactitude = []
# À compléter : boucle bootstrap (rng.integers(0, n, n)), arbre profond, cumul des votes,
# exactitude du vote majoritaire ; comparaison avec un arbre seul et BaggingClassifier
""")
    c.append(md("""### Question 6 — Forêt aléatoire : out-of-bag et importances
Ajuster `RandomForestClassifier(n_estimators=300, oob_score=True)`. Comparer l'exactitude out-of-bag à l'exactitude test, puis afficher les cinq variables les plus importantes."""))
    q(c, corrige, """foret = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=0).fit(Xa, ya)
print("OOB :", round(foret.oob_score_, 3), "| test :", round(foret.score(Xt, yt), 3))
ordre = np.argsort(foret.feature_importances_)[::-1][:5]
for j in ordre:
    print(f"{D.feature_names[j]:25s} {foret.feature_importances_[j]:.3f}")""",
"""# À compléter : RandomForestClassifier(n_estimators=300, oob_score=True, random_state=0),
# oob_score_ contre score test, cinq plus grandes feature_importances_
""")
    c.append(md("""### Question 7 — AdaBoost à la main
Six points $x=1,\\dots,6$, classes $y=(+1,+1,-1,+1,+1,-1)$. Programmer trois tours d'AdaBoost avec des arbres à une coupure (`DecisionTreeClassifier(max_depth=1)` accepte des poids : `fit(x, y, sample_weight=w)`). À chaque tour, afficher la coupure, $\\varepsilon_m$, $\\alpha_m$ et les nouveaux poids ; vérifier que les mal classés pèsent $\\frac12$ après la mise à jour. Comparer avec `AdaBoostClassifier`, dont les poids valent $2\\alpha_m$."""))
    q(c, corrige, """x6 = np.arange(1, 7).reshape(-1, 1); y6 = np.array([1, 1, -1, 1, 1, -1])
w = np.full(6, 1 / 6); F = np.zeros(6)
for m in range(1, 4):
    h = DecisionTreeClassifier(max_depth=1).fit(x6, y6, sample_weight=w)
    pred = h.predict(x6)
    eps = w[pred != y6].sum()
    alpha = 0.5 * np.log((1 - eps) / eps)
    w = w * np.exp(-alpha * y6 * pred); w = w / w.sum()
    F += alpha * pred
    print(f"tour {m} : coupure {h.tree_.threshold[0]}, eps = {eps:.4f}, alpha = {alpha:.3f}, "
          f"poids = {w.round(4)}, mal classés : {w[pred != y6].sum():.2f}")
print("F =", F.round(2), "| prédiction :", np.sign(F).astype(int), "| erreurs :", int(np.sum(np.sign(F) != y6)))

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=3, random_state=0).fit(x6, y6)
print("scikit-learn : poids", ada.estimator_weights_.round(3), "= 2 alpha ; prédiction", ada.predict(x6))""",
"""x6 = np.arange(1, 7).reshape(-1, 1); y6 = np.array([1, 1, -1, 1, 1, -1])
w = np.full(6, 1 / 6); F = np.zeros(6)
# À compléter : trois tours (arbre à une coupure pondéré, eps, alpha, mise à jour et renormalisation des poids, F)
# puis comparaison avec AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=3)
""")
    c.append(md("**Conclusion attendue.** Chaque coupure seule fait au moins une erreur, mais le vote pondéré des trois classe les six points sans erreur. Sur `breast_cancer`, le bagging et la forêt font nettement mieux qu'un arbre seul ; l'erreur out-of-bag donne une bonne idée de l'erreur test sans données mises de côté." if corrige else "**À commenter :** que gagne-t-on à combiner des arbres ? Pourquoi les poids des points mal classés augmentent-ils ?"))
    nb = nbf.v4.new_notebook(); nb["cells"] = c
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb

out = Path(__file__).resolve().parent.parent / "notebooks"
nbf.write(build(False), out / "TP3.ipynb"); nbf.write(build(True), out / "corriges" / "TP3_corrige.ipynb")
print("ok")

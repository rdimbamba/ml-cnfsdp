import nbformat as nbf
from pathlib import Path
md = nbf.v4.new_markdown_cell; co = nbf.v4.new_code_cell
SIMU = open(Path(__file__).with_name("menages.py")).read().replace("import numpy as np, pandas as pd\n", "")

def build(corrige):
    c = []
    def q(sol, squel): c.append(co(sol if corrige else squel))
    t = "TP4 — ACP, k-means, classification hiérarchique" + (" (corrigé)" if corrige else "")
    c.append(md(f"""# {t}
**CNFSDP — Introduction au Machine Learning** · L3 S5 · 2026-2027 · Rois Céti DIMBAMBA · chapitre 4 · 45 min en salle

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/{'corriges/TP4_corrige' if corrige else 'TP4'}.ipynb)

Partie A : ACP à la main puis indice de richesse. Partie B : k-means et choix du nombre de groupes. Partie C : CAH et forêt d'isolement.
Les ménages sont **fictifs** : ils sont simulés au format des biens relevés par les enquêtes DHS, ce qui permet de connaître la « vraie » richesse et de juger l'indice. Aucun téléchargement n'est nécessaire."""))
    c.append(co("""import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
VERT, OR, GRIS = "#126B2A", "#B8860B", "#8A8A8A"   # couleurs du cours
"""))
    c.append(md("Fonction de simulation des ménages (à exécuter sans la modifier) :"))
    c.append(co(SIMU.strip() + "\n\nbiens, richesse, urbain = simuler_menages()\nbiens.head()"))

    c.append(md("""## Partie A — Analyse en composantes principales
### Question 1 — ACP à la main
Quatre ménages, scores centrés de logement et d'équipement : $(-2,-1)$, $(-1,-2)$, $(1,2)$, $(2,1)$. Calculer $S=\\frac14X^\\top X$, ses valeurs et vecteurs propres (`np.linalg.eigh`), puis tracer $u^\\top Su$ pour $u=(\\cos\\theta,\\sin\\theta)$, $\\theta\\in[0,180°]$. Retrouver le maximum $4{,}5$ en $45°$ et la part d'inertie de $90\\,\\%$."""))
    q("""X4 = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]], float)
S = X4.T @ X4 / 4
lam, U = np.linalg.eigh(S)                     # valeurs propres croissantes
print(S); print("valeurs propres :", lam, "| axe 1 :", U[:, -1].round(3))
theta = np.radians(np.arange(0, 181))
var = [np.array([np.cos(t), np.sin(t)]) @ S @ np.array([np.cos(t), np.sin(t)]) for t in theta]
plt.figure(figsize=(5, 2.8)); plt.plot(np.degrees(theta), var, color=VERT)
plt.xlabel("angle (degrés)"); plt.ylabel("variance projetée"); plt.show()
print("maximum", round(max(var), 3), "en", int(np.degrees(theta[np.argmax(var)])), "degrés | part :", lam[-1] / lam.sum())""",
"""X4 = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]], float)
# À compléter : S, np.linalg.eigh(S), tracé de u'Su selon l'angle, maximum et part d'inertie
""")
    c.append(md("""### Question 2 — L'indice de richesse
Faire une ACP normée des 12 biens (`StandardScaler` puis `PCA`). Afficher l'éboulis des valeurs propres et les corrélations de chaque bien avec les deux premières composantes ($\\sqrt{\\lambda_k}\\,u_{jk}$). Construire l'indice de richesse (composante 1, orientée pour croître avec l'électricité), ses quintiles, et le taux d'équipement en réfrigérateur par quintile. Comparer l'indice à la richesse cachée `richesse`."""))
    q("""Z = StandardScaler().fit_transform(biens)
acp = PCA().fit(Z)
lam = acp.explained_variance_
plt.figure(figsize=(5, 2.8)); plt.bar(range(1, 13), lam, color=VERT); plt.axhline(1, color=OR, ls="--")
plt.xlabel("composante"); plt.ylabel("valeur propre"); plt.show()
print("parts d'inertie :", acp.explained_variance_ratio_[:3].round(3))

corr = pd.DataFrame(acp.components_[:2].T * np.sqrt(lam[:2]), index=biens.columns, columns=["axe 1", "axe 2"])
print(corr.round(2))

indice = acp.transform(Z)[:, 0]
if np.corrcoef(indice, biens["electricite"])[0, 1] < 0:
    indice = -indice
quintile = pd.qcut(indice, 5, labels=False) + 1
print("réfrigérateur par quintile :", biens.groupby(quintile)["refrigerateur"].mean().round(2).values)
print("corrélation avec la richesse cachée :", round(np.corrcoef(indice, richesse)[0, 1], 3))""",
"""# À compléter : Z standardisé, PCA().fit(Z), éboulis, corrélations biens-axes,
# indice = composante 1 orientée, quintiles (pd.qcut), réfrigérateur par quintile, corrélation avec richesse
""")
    c.append(md("**À commenter :** combien d'axes garder (règle $\\lambda>1$) ? Que représente l'axe 1 ? Quels biens tirent l'axe 2 ?"))

    c.append(md("""## Partie B — k-means
### Question 3 — L'algorithme de Lloyd en NumPy
Programmer k-means ($k=2$) pour les revenus $1, 2, 4, 6, 9, 12$ : affectation au centre le plus proche, puis centres = moyennes, jusqu'à stabilité, en affichant $W$ à chaque itération. Partir des centres $1$ et $2$, puis $3$ et $11$. Quelle partition a la plus petite inertie ?"""))
    q("""x = np.array([1, 2, 4, 6, 9, 12.])

def lloyd(x, centres):
    centres = np.array(centres, float)
    while True:
        groupe = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)   # affectation
        W = sum(((x[groupe == j] - centres[j]) ** 2).sum() for j in range(len(centres)))
        print("centres", centres.round(2), "| groupes", [x[groupe == j].tolist() for j in range(len(centres))], "| W =", round(W, 2))
        nouveaux = np.array([x[groupe == j].mean() for j in range(len(centres))])   # mise à jour
        if np.allclose(nouveaux, centres):
            return groupe, W
        centres = nouveaux

lloyd(x, [1, 2]); print()
lloyd(x, [3, 11])""",
"""x = np.array([1, 2, 4, 6, 9, 12.])

def lloyd(x, centres):
    # À compléter : boucle affectation / mise à jour, affichage de W, arrêt quand les centres ne bougent plus
    pass

lloyd(x, [1, 2]); print()
lloyd(x, [3, 11])
""")
    c.append(md("""### Question 4 — Combien de groupes de ménages ?
Sur les deux premières composantes, calculer l'inertie intra $W(k)$ et la silhouette moyenne de `KMeans(n_init=10)` pour $k=2,\\dots,8$ ; tracer les deux courbes. Retenir $k=3$, puis décrire chaque groupe par son effectif et ses taux d'équipement (électricité, voiture, bétail)."""))
    q("""C = PCA(n_components=2).fit_transform(Z)
ks = range(2, 9); W, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(C)
    W.append(km.inertia_); sil.append(silhouette_score(C, km.labels_))
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(ks, W, color=VERT, marker="o"); axes[0].set_title("inertie intra W(k)")
axes[1].plot(ks, sil, color=OR, marker="o"); axes[1].set_title("silhouette moyenne"); plt.show()

km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(C)
profil = biens.groupby(km.labels_)[["electricite", "voiture", "betail"]].mean().round(2)
profil["effectif"] = pd.Series(km.labels_).value_counts().sort_index()
print(profil)""",
"""# À compléter : C = deux premières composantes, boucle sur k (inertia_, silhouette_score), deux tracés,
# puis KMeans à 3 groupes et profil des groupes (groupby sur km.labels_)
""")

    c.append(md("""## Partie C — Classification hiérarchique et forêt d'isolement
### Question 5 — CAH de Ward sur les six revenus
Avec `linkage(x.reshape(-1, 1), method="ward")`, tracer le dendrogramme. scipy affiche une hauteur $h=\\sqrt{2\\Delta}$ : retrouver les hausses d'inertie $\\Delta=h^2/2$ du cours ($0{,}5$ ; $2$ ; $4{,}5$ ; $12{,}25$ ; $70{,}08$). Pour la partition en 2 groupes, vérifier $T=W+B$."""))
    q("""L = linkage(x.reshape(-1, 1), method="ward")
dendrogram(L, labels=[str(int(v)) for v in x]); plt.ylabel("hauteur scipy"); plt.show()
print("hausses d'inertie :", (L[:, 2] ** 2 / 2).round(2))

groupes = [x[x <= 6], x[x > 6]]
T = ((x - x.mean()) ** 2).sum()
W = sum(((g - g.mean()) ** 2).sum() for g in groupes)
B = sum(len(g) * (g.mean() - x.mean()) ** 2 for g in groupes)
print(f"T = {T:.2f} | W = {W:.2f} | B = {B:.2f} | W + B = {W + B:.2f} | B/T = {B / T:.0%}")""",
"""L = linkage(x.reshape(-1, 1), method="ward")
# À compléter : dendrogramme, hausses d'inertie L[:, 2]**2 / 2, puis T, W et B pour {1,2,4,6} et {9,12}
""")
    c.append(md("""### Question 6 — Les ménages atypiques
Ajuster `IsolationForest(contamination=0.01)` sur les biens standardisés. Combien de ménages sont signalés ? Comparer leurs taux d'équipement à ceux de l'ensemble des ménages : qu'ont-ils d'inhabituel ?"""))
    q("""iso = IsolationForest(contamination=0.01, random_state=0).fit(Z)
atypiques = biens[iso.predict(Z) == -1]
print(len(atypiques), "ménages atypiques")
print(pd.DataFrame({"atypiques": atypiques.mean(), "tous": biens.mean()}).round(2))""",
"""# À compléter : IsolationForest(contamination=0.01, random_state=0).fit(Z), ménages prédits -1, comparaison des moyennes
""")
    c.append(md("**Conclusion attendue.** La première composante résume l'équipement des ménages et suit de près la richesse cachée : c'est le principe de l'indice de richesse DHS. Lloyd peut s'arrêter sur un minimum local (W = 22,67 contre 19,25) ; la CAH de Ward retrouve ici la meilleure partition. Les ménages atypiques combinent des biens rarement possédés ensemble." if corrige else "**Pour aller plus loin :** refaire l'ACP et une CAH sur des indicateurs des pays africains (Banque mondiale, World Development Indicators, https://data.worldbank.org), par exemple PIB par habitant, espérance de vie, scolarisation et accès à l'électricité."))
    nb = nbf.v4.new_notebook(); nb["cells"] = c
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb

out = Path(__file__).resolve().parent.parent / "notebooks"
nbf.write(build(False), out / "TP4.ipynb"); nbf.write(build(True), out / "corriges" / "TP4_corrige.ipynb")
print("ok")

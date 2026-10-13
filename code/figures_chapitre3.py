"""Figures du chapitre 3 (Classification et méthodes d'ensemble) -- CNFSDP 2026-2027. Sorties dans ../cours/chapitre3/."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier

VERT, OR, GRIS, ENCRE = "#126B2A", "#B8860B", "#8A8A8A", "#1E1E1E"
virgule = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
def fr(ax):
    ax.xaxis.set_major_formatter(virgule); ax.yaxis.set_major_formatter(virgule)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": ENCRE,
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})
OUT = "../cours/chapitre3/"
FOND = ListedColormap(["#DCEBDF", "#F3E7C6"])          # zones prédites 0 (vert pâle) et 1 (or pâle)

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
Xt, yt = make_moons(n_samples=5000, noise=0.3, random_state=1)
gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 300), np.linspace(-1.4, 1.9, 300))
G = np.c_[gx.ravel(), gy.ravel()]
def panneau(ax, f, titre):
    f.fit(X, y); Z = f.predict(G).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], cmap=FOND)
    ax.contour(gx, gy, Z, levels=[0.5], colors=ENCRE, linewidths=0.8)
    ax.scatter(*X[y == 0].T, s=6, color=VERT, lw=0); ax.scatter(*X[y == 1].T, s=7, color=OR, marker="^", lw=0)
    err = np.mean(f.predict(Xt) != yt)
    ax.set_title(f"{titre}\nerreur test {err:.1%}".replace(".", ","), fontsize=8.5, color=ENCRE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    return err

# --- Figure 1 : k plus proches voisins ---
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3))
for ax, k in zip(axs, [1, 15, 150]):
    e = panneau(ax, KNeighborsClassifier(n_neighbors=k), f"$k={k}$"); print("knn", k, round(e, 4))
fig.tight_layout(); fig.savefig(OUT + "knn.pdf"); plt.close(fig)

# --- Figure 2 : sigmoïde et régression logistique sur le fil rouge ---
sig = lambda t: 1 / (1 + np.exp(-t))
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))
t = np.linspace(-6, 6, 300); ax = axs[0]
ax.plot(t, sig(t), color=VERT, lw=2); ax.axhline(0.5, color=GRIS, ls="--", lw=1); ax.axvline(0, color="#DDDDDD", lw=0.8)
ax.text(-5.8, 0.56, r"$\sigma(0)=\frac{1}{2}$", color="#555555", fontsize=8)
ax.text(1.4, 0.2, r"$\sigma(t)=\dfrac{1}{1+e^{-t}}$", color=VERT, fontsize=9)
ax.set_xlabel("$t$"); ax.set_ylim(-0.05, 1.05); fr(ax)
etudes = np.array([0, 1, 3, 4, 6, 6, 7, 9, 10, 2]); vac = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0])
ax = axs[1]; xx = np.linspace(-0.5, 12, 300)
ax.plot(xx, sig(-1.46 + 0.43 * xx), color=VERT, lw=2)
dec = np.where((etudes == 6) & (vac == 1), 0.25, 0)
ax.scatter(etudes + dec, vac, s=26, color=[OR if v else VERT for v in vac], zorder=3,
           marker="o")
ax.axhline(0.5, color=GRIS, ls="--", lw=1); ax.axvline(1.46 / 0.43, color=GRIS, ls=":", lw=1)
ax.text(3.6, 0.08, "frontière\n$x\\approx3{,}4$ ans", fontsize=7.5, color="#555555")
ax.text(6.8, 0.62, r"$\hat\eta(x)=\sigma(-1{,}46+0{,}43\,x)$", fontsize=8.5, color=VERT)
ax.set_xlabel("années d'études de la mère $x$"); ax.set_ylabel("$y$ (vacciné) et $\\hat\\eta(x)$")
ax.set_ylim(-0.08, 1.08); fr(ax)
fig.tight_layout(); fig.savefig(OUT + "sigmoide.pdf"); plt.close(fig)

# --- Figure 3 : frontières de quatre classifieurs ---
fig, axs = plt.subplots(1, 4, figsize=(6.8, 2.0))
for ax, (f, tt) in zip(axs, [(LogisticRegression(), "logistique"), (DecisionTreeClassifier(random_state=0), "arbre profond"),
                            (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0), "forêt aléatoire"),
                            (GradientBoostingClassifier(random_state=0), "gradient boosting")]):
    e = panneau(ax, f, tt); print(tt, round(e, 4))
fig.tight_layout(); fig.savefig(OUT + "frontieres.pdf"); plt.close(fig)

# --- Figure 4 : erreur selon le nombre d'arbres ---
B = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300]
graines = range(8)
eb = [np.mean([np.mean(BaggingClassifier(DecisionTreeClassifier(), n_estimators=b, random_state=r).fit(X, y).predict(Xt) != yt)
               for r in graines]) for b in B]
ef = [np.mean([np.mean(RandomForestClassifier(n_estimators=b, max_features=1, random_state=r).fit(X, y).predict(Xt) != yt)
               for r in graines]) for b in B]
arbre = np.mean([np.mean(DecisionTreeClassifier(random_state=r).fit(X, y).predict(Xt) != yt) for r in graines])
fig, ax = plt.subplots(figsize=(4.2, 2.4))
ax.plot(B, eb, color=OR, lw=2, marker="o", ms=3.5); ax.plot(B, ef, color=VERT, lw=2, marker="s", ms=3.5)
ax.axhline(arbre, color=GRIS, ls="--", lw=1)
ax.text(330, eb[-1] + 0.002, "bagging", va="bottom", fontsize=8); ax.text(330, ef[-1] - 0.002, "forêt", va="top", fontsize=8)
ax.text(1.1, arbre + 0.003, "un seul arbre profond", fontsize=7.5, color="#555555")
ax.set_xscale("log"); ax.set_xlim(0.9, 1500); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
ax.set_xlabel("nombre d'arbres $B$ (échelle log)"); ax.set_ylabel("taux d'erreur test")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}".replace(".", ",")))
fig.tight_layout(); fig.savefig(OUT + "nb_arbres.pdf"); plt.close(fig)
print("bagging", np.round(eb, 3), "\nforet", np.round(ef, 3), "\narbre", round(arbre, 3))

# --- Figure 5 : AdaBoost pas à pas sur six points ---
x = np.arange(1, 7); yy = np.array([1, 1, -1, 1, 1, -1])
poids = [np.full(6, 1/6), np.array([.1, .1, .5, .1, .1, .1]), np.array([.0625, .0625, .3125, .25, .25, .0625])]
seuils = [(5.5, "$+1$ si $x\\leq5{,}5$"), (2.5, "$+1$ si $x\\leq2{,}5$"), (3.5, "$-1$ si $x\\leq3{,}5$")]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.7), sharey=True)
for m, ax in enumerate(axs):
    w = poids[m]
    for xi, yi, wi in zip(x, yy, w):
        ax.scatter(xi, 0, s=1400 * wi, color=OR if yi == 1 else VERT, marker="^" if yi == 1 else "o", zorder=3)
        ax.text(xi, -0.62, f"{wi:.2f}".replace(".", ","), ha="center", fontsize=7, color="#555555")
    ax.axvline(seuils[m][0], color=ENCRE, lw=1.2)
    ax.set_title(f"tour {m+1} : " + seuils[m][1], fontsize=8.5, color=ENCRE)
    ax.set_ylim(-0.8, 0.6); ax.set_xlim(0.4, 6.6); ax.set_yticks([]); ax.set_xticks(x)
    ax.spines["left"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "adaboost.pdf"); plt.close(fig)

"""Figures du chapitre 2 (Régression) -- CNFSDP 2026-2027. Sorties dans ../cours/chapitre2/."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, lasso_path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

VERT, OR, GRIS, ENCRE = "#126B2A", "#B8860B", "#8A8A8A", "#1E1E1E"
virgule = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
def fr(ax, x=True, y=True):
    if x: ax.xaxis.set_major_formatter(virgule)
    if y: ax.yaxis.set_major_formatter(virgule)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": ENCRE,
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})
OUT = "../cours/chapitre2/"

# --- Figure 1 : droite des moindres carrés et résidus (fil rouge) ---
x = np.array([0, 3, 6, 9, 12.]); y = np.array([6, 5, 5, 3, 1.])
a, b = 6.4, -0.4
fig, ax = plt.subplots(figsize=(3.1, 2.4))
xx = np.array([-0.8, 12.8]); ax.plot(xx, a + b*xx, color=VERT, lw=2)
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [yi, a + b*xi], color=OR, lw=1.4)
ax.scatter(x, y, s=28, color=ENCRE, zorder=3)
ax.text(6.6, 5.55, r"$\hat y=6{,}4-0{,}4\,x$", color=VERT, fontsize=9)
ax.annotate("résidu $y_i-\\hat y_i$", xy=(6, 4.5), xytext=(0.2, 1.6), fontsize=8, color=OR,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.set_xlabel("années d'études de la mère $x$"); ax.set_ylabel("nombre d'enfants $y$")
ax.set_xlim(-1, 13); ax.set_ylim(0, 7); ax.set_xticks([0, 3, 6, 9, 12])
fr(ax); fig.tight_layout(); fig.savefig(OUT + "droite.pdf"); plt.close(fig)

# --- Figure 2 : sur-apprentissage d'une régression polynomiale ---
rng = np.random.default_rng(3)
g = lambda t: np.sin(2*np.pi*t)
def simuler(n):
    t = rng.uniform(0, 1, n); return t, g(t) + 0.3*rng.standard_normal(n)
def poly(d, lam=0.0):
    reg = LinearRegression() if lam == 0 else Ridge(alpha=lam)
    return make_pipeline(PolynomialFeatures(d), StandardScaler(), reg)
t, yt = simuler(20)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax = axs[0]; grille = np.linspace(0, 1, 400)
ax.scatter(t, yt, s=14, color=ENCRE, zorder=3)
for d, c, lab in [(1, GRIS, "degré 1"), (3, VERT, "degré 3"), (15, OR, "degré 15")]:
    f = poly(d).fit(t[:, None], yt); ax.plot(grille, f.predict(grille[:, None]), color=c, lw=1.8, label=lab)
ax.legend(frameon=False, fontsize=7.5, loc="lower left", handlelength=1.5)
ax.set_ylim(-2, 2.2); ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); fr(ax)
ax.set_title("$n=20$ points", fontsize=9, color=ENCRE)
degres = np.arange(1, 16); R = 200
tr, te = np.zeros((R, 15)), np.zeros((R, 15))
for r in range(R):
    t, yy = simuler(20); tt, yyt = simuler(2000)
    for j, d in enumerate(degres):
        f = poly(d).fit(t[:, None], yy)
        tr[r, j] = np.mean((f.predict(t[:, None]) - yy)**2); te[r, j] = np.mean((f.predict(tt[:, None]) - yyt)**2)
tr, te = np.median(tr, 0), np.median(te, 0)
ax = axs[1]
ax.plot(degres, te, color=OR, lw=2, marker="o", ms=3.5); ax.plot(degres, tr, color=VERT, lw=2, marker="s", ms=3.5)
ax.axhline(0.09, color=GRIS, lw=1.1, ls="--"); ax.text(7.5, 0.1, r"bruit $\sigma^2=0{,}09$", color="#555555", fontsize=7.5)
ax.text(15.3, te[-1], "test", fontsize=8, va="center"); ax.text(15.3, tr[-1], "apprentissage", fontsize=8, va="center")
ax.set_yscale("log"); ax.set_ylim(0.01, 3); ax.set_xlim(0.5, 19.5)
ax.set_xlabel("degré du polynôme"); ax.set_ylabel("erreur quadratique (médiane)")
ax.set_yticks([0.01, 0.1, 1]); fr(ax)
fig.tight_layout(); fig.savefig(OUT + "polynome.pdf"); plt.close(fig)
print("degré optimal", degres[te.argmin()], "test", te.round(3))

# --- Figure 3 : chemins de régularisation Ridge et Lasso (diabète, variables standardisées) ---
D = load_diabetes(as_frame=True); X = StandardScaler().fit_transform(D.data); y = D.target.values
n = len(y); noms = list(D.data.columns)
lams = np.logspace(-2, 4, 80)
coef_r = np.array([Ridge(alpha=n*l).fit(X, y).coef_ for l in lams])           # lambda du cours = alpha / n
alph, coef_l, _ = lasso_path(X, y, alphas=np.logspace(-2, 2, 80)); coef_l = coef_l.T; lam_l = 2*alph  # lambda = 2 alpha
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for ax, L, C, titre in [(axs[0], lams, coef_r, "Ridge"), (axs[1], lam_l, coef_l, "Lasso")]:
    for j, nm in enumerate(noms):
        c, lw = (VERT, 2) if nm == "bmi" else (OR, 2) if nm == "s5" else (GRIS, 0.9)
        ax.plot(L, C[:, j], color=c, lw=lw)
    ax.set_xscale("log"); ax.axhline(0, color="#CCCCCC", lw=0.6)
    ax.set_title(titre, fontsize=9, color=ENCRE); ax.set_xlabel(r"pénalité $\lambda$ (échelle log)")
    i0 = np.argmin(L)
    ax.text(L[i0]*1.3, C[i0, noms.index("bmi")]-9, "IMC (bmi)", color=VERT, fontsize=7.5)
    ax.text(L[i0]*1.3, C[i0, noms.index("s5")]+3, "s5", color=OR, fontsize=7.5)
    fr(ax, x=False)
axs[0].set_ylabel("coefficient")
fig.tight_layout(); fig.savefig(OUT + "chemins.pdf"); plt.close(fig)
print("Lasso: nb non nuls", [(round(l, 2), int((c != 0).sum())) for l, c in zip(lam_l[::16], coef_l[::16])])

# --- Figure 4 : un arbre, une forêt, le boosting ---
rng = np.random.default_rng(0)
t = rng.uniform(0, 1, 150); yy = g(t) + 0.3*rng.standard_normal(150)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
ax = axs[0]; ax.scatter(t, yy, s=7, color="#BBBBBB", lw=0)
arbre = DecisionTreeRegressor(random_state=0).fit(t[:, None], yy)
foret = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0).fit(t[:, None], yy)
ax.plot(grille, arbre.predict(grille[:, None]), color=OR, lw=1.1)
ax.plot(grille, foret.predict(grille[:, None]), color=VERT, lw=2.2)
ax.text(0.62, 1.5, "un arbre profond", color=OR, fontsize=8); ax.text(0.05, -1.65, "forêt de 300 arbres", color=VERT, fontsize=8)
ax.set_title("Bagging : moyenne d'arbres", fontsize=9, color=ENCRE); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax = axs[1]; ax.scatter(t, yy, s=7, color="#BBBBBB", lw=0)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0).fit(t[:, None], yy)
etapes = list(gb.staged_predict(grille[:, None]))
ax.plot(grille, etapes[4], color=OR, lw=1.8); ax.plot(grille, etapes[199], color=VERT, lw=2.2)
ax.text(0.55, 1.5, "après 5 petits arbres", color=OR, fontsize=8); ax.text(0.05, -1.65, "après 200 petits arbres", color=VERT, fontsize=8)
ax.set_title("Boosting : corrections successives", fontsize=9, color=ENCRE); ax.set_xlabel("$x$")
for ax in axs: ax.set_ylim(-2, 2); fr(ax)
fig.tight_layout(); fig.savefig(OUT + "ensembles.pdf"); plt.close(fig)

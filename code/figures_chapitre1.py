import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

VERT, OR, GRIS, ENCRE = "#126B2A", "#B8860B", "#8A8A8A", "#1E1E1E"
from matplotlib.ticker import FuncFormatter
virgule = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
def fr(ax):
    ax.xaxis.set_major_formatter(virgule); ax.yaxis.set_major_formatter(virgule)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": ENCRE,
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})

# --- Figure 1 : sur-apprentissage d'un arbre (régression) ---
rng = np.random.default_rng(0)
sigma, n, m, R = 0.3, 200, 5_000, 30
prof = np.arange(1, 16)
tr, te = np.zeros((R, len(prof))), np.zeros((R, len(prof)))
for r in range(R):
    X = rng.uniform(0, 1, (n, 1)); y = np.sin(2*np.pi*X[:, 0]) + sigma*rng.standard_normal(n)
    Xt = rng.uniform(0, 1, (m, 1)); yt = np.sin(2*np.pi*Xt[:, 0]) + sigma*rng.standard_normal(m)
    for j, p in enumerate(prof):
        f = DecisionTreeRegressor(max_depth=p, random_state=0).fit(X, y)
        tr[r, j] = np.mean((f.predict(X) - y)**2); te[r, j] = np.mean((f.predict(Xt) - yt)**2)
tr, te = tr.mean(0), te.mean(0)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(prof, te, color=OR, lw=2, marker="o", ms=4, label="risque test")
ax.plot(prof, tr, color=VERT, lw=2, marker="s", ms=4, ls="-", label="risque d'apprentissage")
ax.axhline(sigma**2, color=GRIS, lw=1.2, ls="--")
ax.text(9.5, sigma**2 - 0.022, r"risque de Bayes $R^*=\sigma^2$", va="center", ha="left", color="#555555", fontsize=7.5)
ax.text(prof[-1]+0.2, te[-1], "test", color=ENCRE, va="center", fontsize=8)
ax.text(prof[-1]+0.2, tr[-1], "apprentissage", color=ENCRE, va="center", fontsize=8)
best = prof[te.argmin()]
ax.annotate(f"optimum : profondeur {best}", xy=(best, te.min()), xytext=(best+2.5, 0.3),
            fontsize=8, color=ENCRE, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.set_xlabel("profondeur maximale de l'arbre"); ax.set_ylabel("erreur quadratique moyenne")
ax.set_xlim(0.5, 18.5); ax.set_ylim(0, 0.45); ax.grid(axis="y", color="#EEEEEE", lw=0.6)
fr(ax); fig.tight_layout(); fig.savefig("../cours/chapitre1/surapprentissage.pdf"); plt.close(fig)
print("profondeur optimale", best, "test min", te.min().round(4), "train@15", tr[-1].round(4))

# --- Figure 2 : mélange gaussien et frontière de Bayes ---
rng = np.random.default_rng(1)
y = rng.integers(0, 2, 400); X = rng.standard_normal((400, 2)) + 1.5*y[:, None]
fig, ax = plt.subplots(figsize=(3.0, 2.6))
ax.scatter(*X[y == 0].T, s=10, color=VERT, marker="o", alpha=0.75, lw=0, label="$Y=0$")
ax.scatter(*X[y == 1].T, s=12, color=OR, marker="^", alpha=0.75, lw=0, label="$Y=1$")
xx = np.linspace(-3, 4.5, 2); ax.plot(xx, 1.5 - xx, color=ENCRE, lw=1.5)
ax.annotate(r"frontière de Bayes $x_1+x_2=1{,}5$", xy=(3.9, -2.4), xytext=(-2.9, -2.85), fontsize=7.5, color=ENCRE,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.2, markerscale=1.8)
fr(ax); fig.tight_layout(); fig.savefig("../cours/chapitre1/melange.pdf"); plt.close(fig)

# --- Figure 3 : courbe ROC ---
rng = np.random.default_rng(0)
y = rng.integers(0, 2, 3_000); X = rng.standard_normal((3_000, 2)) + 1.5*y[:, None]
f = LogisticRegression().fit(X[:1000], y[:1000]); s = f.predict_proba(X[1000:])[:, 1]
fpr, tpr, _ = roc_curve(y[1000:], s); auc = roc_auc_score(y[1000:], s)
fig, ax = plt.subplots(figsize=(2.7, 2.5))
ax.fill_between(fpr, tpr, color=VERT, alpha=0.10, lw=0)
ax.plot(fpr, tpr, color=VERT, lw=2)
ax.plot([0, 1], [0, 1], color=GRIS, lw=1, ls="--")
ax.text(0.40, 0.30, f"AUC = {auc:.3f}".replace(".", ","), fontsize=8, color=ENCRE)
ax.text(0.66, 0.52, "hasard", fontsize=7.5, color="#666666", rotation=38)
ax.set_xlabel("taux de faux positifs"); ax.set_ylabel("taux de vrais positifs")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
fr(ax); fig.tight_layout(); fig.savefig("../cours/chapitre1/roc.pdf"); plt.close(fig)
print("AUC", round(auc, 4))

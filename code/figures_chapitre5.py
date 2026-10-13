"""Figures du chapitre 5 (Projet de bout en bout, interprétabilité, éthique) -- CNFSDP 2026-2027."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance, partial_dependence
from enfants_dhs import simuler_enfants

VERT, OR, GRIS, ENCRE = "#126B2A", "#B8860B", "#8A8A8A", "#1E1E1E"
virgule = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
pourcent = FuncFormatter(lambda v, p: f"{v:.0%}".replace("%", " %"))
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": ENCRE,
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})
OUT = "../cours/chapitre5/"

df = simuler_enfants()
df["m14"] = df["m14"].replace(98, np.nan)
qual = ["v024", "v025", "b4", "m15", "v467d"]; quant = ["v106", "v190", "bord", "hw1", "m14"]
df[quant] = df[quant].astype(float)
X, y = df[qual + quant], df["vacc_complet"]
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), qual),
                               (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), quant))
f = make_pipeline(prep, LogisticRegression(max_iter=1000)).fit(Xa, ya)

# --- Figure 1 : importance par permutation et dépendance partielle ---
noms = {"v190": "quintile de richesse", "v467d": "distance (problème)", "m15": "lieu d'accouchement",
        "v024": "région", "v106": "instruction de la mère", "m14": "visites prénatales", "v025": "milieu",
        "bord": "rang de naissance", "hw1": "âge (mois)", "b4": "sexe"}
r = permutation_importance(f, Xt, yt, scoring="roc_auc", n_repeats=20, random_state=0)
imp = pd.Series(r.importances_mean, index=X.columns).sort_values()
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
ax.barh([noms[c] for c in imp.index], imp.values, color=[VERT if v > 0.01 else GRIS for v in imp.values], height=0.65)
ax.set_xlabel("baisse de l'AUC test quand on mélange la variable"); ax.xaxis.set_major_formatter(virgule)
ax.tick_params(axis="y", labelsize=7.5)
ax = axs[1]
for var, c, lab in [("v106", VERT, "instruction de la mère\n(0 aucune ... 3 supérieur)"), ("v190", OR, "quintile de richesse (1 à 5)")]:
    pdp = partial_dependence(f, Xa, [var], kind="average", grid_resolution=5)
    ax.plot(pdp["grid_values"][0], pdp["average"][0], color=c, marker="o", lw=2, ms=4, label=lab)
ax.set_ylabel("probabilité moyenne prédite\nde vaccination complète"); ax.set_xlabel("valeur imposée à la variable")
ax.yaxis.set_major_formatter(pourcent); ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_ylim(0.3, 0.85)
fig.tight_layout(); fig.savefig(OUT + "interpretation.pdf"); plt.close(fig)
print(imp.sort_values(ascending=False).round(3).to_dict())

# --- Figure 2 : équité selon le milieu (positif = enfant non vacciné, signalé pour une relance) ---
p_non = 1 - f.predict_proba(Xt)[:, 1]; signal = p_non > 0.5; vrai = (yt == 0).values
res = {}
for g, nm in [(1, "urbain"), (2, "rural")]:
    m = (Xt["v025"] == g).values
    vp = (signal & vrai & m).sum(); fp = (signal & ~vrai & m).sum(); fn = (~signal & vrai & m).sum(); vn = (~signal & ~vrai & m).sum()
    res[nm] = {"enfants signalés": signal[m].mean(), "rappel": vp / (vp + fn), "précision": vp / (vp + fp)}
    print(nm, vp, fp, fn, vn)
R = pd.DataFrame(res)
fig, ax = plt.subplots(figsize=(4.4, 2.3))
xk = np.arange(3)
ax.bar(xk - 0.18, R["urbain"], width=0.34, color=VERT, label="urbain")
ax.bar(xk + 0.18, R["rural"], width=0.34, color=OR, label="rural")
for i, (u, ru) in enumerate(zip(R["urbain"], R["rural"])):
    ax.text(i - 0.18, u + 0.02, f"{u:.0%}".replace("%", " %"), ha="center", fontsize=7.5)
    ax.text(i + 0.18, ru + 0.02, f"{ru:.0%}".replace("%", " %"), ha="center", fontsize=7.5)
ax.set_xticks(xk); ax.set_xticklabels(R.index); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(pourcent)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(OUT + "equite.pdf"); plt.close(fig)
print(R.round(3))

# --- Figure 3 : mécanisme de Laplace ---
t = np.linspace(0, 75, 600)
fig, ax = plt.subplots(figsize=(4.4, 2.2))
for eps, c, lab in [(1, VERT, r"$\varepsilon=1$ : bruit d'écart type $1{,}4$"), (0.1, OR, r"$\varepsilon=0{,}1$ : bruit d'écart type $14$")]:
    ax.plot(t, eps / 2 * np.exp(-eps * np.abs(t - 37)), color=c, lw=2, label=lab)
ax.axvline(37, color=GRIS, ls="--", lw=1); ax.text(35.5, 0.3, "vrai nombre : 37", fontsize=7.5, color="#555555", ha="right")
ax.set_xlabel("nombre publié"); ax.set_ylabel("densité"); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_ylim(0, 0.72); ax.xaxis.set_major_formatter(virgule); ax.yaxis.set_major_formatter(virgule)
fig.tight_layout(); fig.savefig(OUT + "laplace.pdf"); plt.close(fig)

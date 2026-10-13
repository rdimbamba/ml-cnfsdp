"""Figures du chapitre 4 (Apprentissage non supervisé) -- CNFSDP 2026-2027. Sorties dans ../cours/chapitre4/."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from menages import simuler_menages

VERT, OR, GRIS, ENCRE = "#126B2A", "#B8860B", "#8A8A8A", "#1E1E1E"
TROIS = [VERT, OR, "#5B6770"]
virgule = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
def fr(ax):
    ax.xaxis.set_major_formatter(virgule); ax.yaxis.set_major_formatter(virgule)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": ENCRE,
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})
OUT = "../cours/chapitre4/"

# --- Figure 1 : ACP à la main (4 ménages) ---
X = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]], float)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
ax = axs[0]
t = np.linspace(-3.2, 3.2, 2); ax.plot(t, t, color=VERT, lw=1.8); ax.plot(t, -t, color=GRIS, lw=1, ls="--")
for p in X:
    q = (p @ np.array([1, 1]) / 2) * np.array([1, 1]); ax.plot([p[0], q[0]], [p[1], q[1]], color=OR, lw=1)
ax.scatter(*X.T, s=30, color=ENCRE, zorder=3)
for i, p in enumerate(X):
    dx, dy = (-0.45, 0.15) if i in (0, 2) else (0.15, -0.5)
    ax.text(p[0] + dx, p[1] + dy, f"{i+1}", fontsize=8)
ax.text(-0.3, 2.75, "axe 1 : $u_1$", color=VERT, fontsize=8.5); ax.text(-3.0, 2.0, "axe 2 : $u_2$", color="#555555", fontsize=8.5)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
ax.set_xlabel("$x_1$ : logement (centré)"); ax.set_ylabel("$x_2$ : équipement (centré)"); fr(ax)
ax = axs[1]; th = np.linspace(0, 180, 361)
ax.plot(th, 2.5 + 2 * np.sin(np.radians(2 * th)), color=VERT, lw=2)
ax.axhline(4.5, color=GRIS, ls="--", lw=1); ax.axhline(0.5, color=GRIS, ls="--", lw=1)
ax.scatter([45], [4.5], color=OR, zorder=3, s=25); ax.scatter([135], [0.5], color=GRIS, zorder=3, s=20)
ax.text(52, 4.62, r"maximum $\lambda_1=4{,}5$ en $\theta=45°$", fontsize=8)
ax.text(20, 0.12, r"minimum $\lambda_2=0{,}5$ en $\theta=135°$", fontsize=8, color="#555555")
ax.set_xlabel(r"angle $\theta$ de la direction $u$ (degrés)"); ax.set_ylabel(r"variance projetée $u^\top Su$")
ax.set_xticks([0, 45, 90, 135, 180]); ax.set_ylim(0, 5.2); fr(ax)
fig.tight_layout(); fig.savefig(OUT + "acp_main.pdf"); plt.close(fig)

# --- Figure 2 : éboulis et cercle des corrélations (ménages simulés) ---
df, richesse, urbain = simuler_menages()
Z = StandardScaler().fit_transform(df); acp = PCA().fit(Z); C = acp.transform(Z)
if np.corrcoef(C[:, 0], df["electricite"])[0, 1] < 0: C[:, 0] *= -1; acp.components_[0] *= -1
lam = acp.explained_variance_; part = acp.explained_variance_ratio_
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]; k = np.arange(1, 13)
ax.bar(k, lam, color=[VERT] + [GRIS] * 11, width=0.7)
ax.axhline(1, color=OR, ls="--", lw=1); ax.text(8.2, 1.15, "seuil $\\lambda=1$", color=OR, fontsize=8)
ax.text(1.5, lam[0] - 0.2, f"{part[0]:.0%} de l'inertie".replace("%", " %"), fontsize=8, color=VERT)
ax.set_xticks(k); ax.set_xlabel("composante"); ax.set_ylabel("valeur propre $\\lambda_k$"); fr(ax)
ax = axs[1]
L = acp.components_[:2].T * np.sqrt(lam[:2])
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#BBBBBB", lw=0.8))
ax.axhline(0, color="#DDDDDD", lw=0.6); ax.axvline(0, color="#DDDDDD", lw=0.6)
noms = {"electricite": "électricité", "eau_courante": "eau courante", "toilettes": "toilettes", "sol_fini": "sol fini",
        "radio": "radio", "television": "télévision", "refrigerateur": "réfrigérateur", "telephone": "téléphone",
        "moto": "moto", "voiture": "voiture", "terre_agricole": "terre agricole", "betail": "bétail"}
montre = {"electricite": (0.04, -0.12), "refrigerateur": (0.04, 0.03), "voiture": (0.04, 0.02), "moto": (0.04, 0.0),
          "radio": (-0.02, 0.07), "terre_agricole": (-0.04, -0.02), "betail": (-0.04, 0.03)}
for (b, nom), (a1, a2) in zip(noms.items(), L):
    c = OR if b in ("terre_agricole", "betail", "moto") else VERT
    ax.annotate("", xy=(a1, a2), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, mutation_scale=7))
    if b in montre:
        dx, dy = montre[b]
        ax.text(a1 + dx, a2 + dy, nom, fontsize=7, color=ENCRE, ha="left" if a1 >= 0 else "right")
ax.text(0.12, -0.55, "et, groupés autour de\nl'électricité : eau courante,\ntoilettes, sol fini,\ntélévision, téléphone",
        fontsize=6.3, color="#555555", va="top")
ax.set_xlim(-1.1, 1.25); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
ax.set_xlabel(f"composante 1 ({part[0]:.0%})".replace("%", " %")); ax.set_ylabel(f"composante 2 ({part[1]:.0%})".replace("%", " %"))
fr(ax); fig.tight_layout(); fig.savefig(OUT + "eboulis_cercle.pdf"); plt.close(fig)
print("lambda", lam[:4].round(2), "part", part[:3].round(3), "corr PC1-richesse", np.corrcoef(C[:, 0], richesse)[0, 1].round(3))

# --- Figure 3 : algorithme de Lloyd en 2D ---
Xb, _ = make_blobs(n_samples=240, centers=[[0, 0], [4, 1], [1.5, 4]], cluster_std=0.9, random_state=2)
c = np.array([[0.5, 0.5], [1.0, 1.5], [1.5, 0.2]])
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.2))
etapes = {0: "départ", 1: "après 1 itération", 9: "convergence"}
it = 0
for ax, cible in zip(axs, [0, 1, 9]):
    while it < cible:
        lab = np.argmin(((Xb[:, None] - c[None]) ** 2).sum(2), 1)
        c = np.array([Xb[lab == k].mean(0) for k in range(3)]); it += 1
    lab = np.argmin(((Xb[:, None] - c[None]) ** 2).sum(2), 1)
    W = ((Xb - c[lab]) ** 2).sum()
    for k in range(3):
        ax.scatter(*Xb[lab == k].T, s=6, color=TROIS[k], lw=0, alpha=0.8)
    ax.scatter(*c.T, s=90, marker="X", color="white", edgecolor=ENCRE, lw=1.2, zorder=3)
    ax.set_title(f"{etapes[cible]} : $W={W:.0f}$", fontsize=8.5, color=ENCRE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "lloyd.pdf"); plt.close(fig)

# --- Figure 4 : coude et silhouette ---
ks = range(1, 9); W = []; S = []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Xb); W.append(km.inertia_)
    S.append(silhouette_score(Xb, km.labels_) if k > 1 else np.nan)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.2))
ax = axs[0]; ax.plot(ks, W, color=VERT, marker="o", lw=2, ms=4); ax.annotate("coude en $k=3$", xy=(3, W[2]), xytext=(4.3, W[1]),
    fontsize=8, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.set_xlabel("nombre de groupes $k$"); ax.set_ylabel("inertie intra $W(k)$"); fr(ax)
ax = axs[1]; ax.plot(ks, S, color=OR, marker="o", lw=2, ms=4); ax.set_xlabel("nombre de groupes $k$"); ax.set_ylabel("silhouette moyenne")
ax.annotate("maximum en $k=3$", xy=(3, S[2]), xytext=(4.5, S[2] - 0.02), fontsize=8, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
fr(ax); fig.tight_layout(); fig.savefig(OUT + "coude.pdf"); plt.close(fig)
print("W", np.round(W, 0), "S", np.round(S, 3))

# --- Figure 5 : dendrogramme de l'exemple à six ménages ---
x6 = np.array([1, 2, 4, 6, 9, 12.]).reshape(-1, 1)
Lk = linkage(x6, method="ward")
Lk[:, 2] = Lk[:, 2] ** 2 / 2                       # hauteur = hausse d'inertie (scipy donne sqrt(2 x hausse))
fig, ax = plt.subplots(figsize=(4.2, 2.4))
dendrogram(Lk, labels=["1", "2", "4", "6", "9", "12"], color_threshold=30, above_threshold_color=GRIS,
           link_color_func=lambda k: VERT, ax=ax)
ax.axhline(30, color=OR, ls="--", lw=1); ax.text(0.01, 36, "coupure : 2 groupes", color=OR, fontsize=8, ha="left", transform=ax.get_yaxis_transform())
ax.set_yscale("symlog", linthresh=1); ax.set_ylabel("hausse d'inertie (échelle log)"); ax.set_xlabel("revenu des ménages")
ax.set_yticks([0, 1, 10, 70]); ax.set_yticklabels(["0", "1", "10", "70"])
fig.tight_layout(); fig.savefig(OUT + "dendrogramme.pdf"); plt.close(fig)
print("hauteurs Ward (hausse d'inertie)", Lk[:, 2].round(3))

# --- Figure 6 : forêt d'isolement ---
rng = np.random.default_rng(4)
Xn = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 300)
Xa = np.array([[3.2, -2.5], [-3, 2.6], [4, 4.2], [-0.5, -3.6], [2.8, 0.1]])
Xi = np.vstack([Xn, Xa])
iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=0).fit(Xi)
out = iso.predict(Xi) == -1
fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax.scatter(*Xi[~out].T, s=8, color=VERT, lw=0, alpha=0.75)
ax.scatter(*Xi[out].T, s=34, color=OR, marker="^", lw=0)
ax.text(-3.9, 4.2, "triangles : jugés atypiques", fontsize=7.5, color=OR)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_xlim(-4, 4.6); ax.set_ylim(-4.3, 4.6); fr(ax)
fig.tight_layout(); fig.savefig(OUT + "isolement.pdf"); plt.close(fig)
print("atypiques", out.sum())

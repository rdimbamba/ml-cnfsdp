import nbformat as nbf
from pathlib import Path
S = {k: Path(__file__).with_name(f"{k}.py").read_text().strip() for k in ["p1","p2","p3","p4"]}
md = nbf.v4.new_markdown_cell; co = nbf.v4.new_code_cell

def build(corrige):
    c = []
    t = "TP0 — Premiers pas" + (" (corrigé)" if corrige else "")
    c.append(md(f"""# {t}
**CNFSDP — Introduction au Machine Learning** · L3 S5 · 2026-2027 · Rois Céti DIMBAMBA

Ce notebook accompagne le **chapitre 0** (polycopié). Il vérifie votre installation, reproduit les programmes de la section 6, puis propose quatre exercices.

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/{'corriges/TP0_corrige' if corrige else 'TP0'}.ipynb)"""))
    c.append(md("## 0. Vérifier l'installation"))
    c.append(co("""import sys
import numpy as np, pandas as pd, sklearn
print("Python", sys.version.split()[0], "| NumPy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__)"""))
    titres = ["Matrice de covariance et valeurs propres", "Droite des moindres carrés et descente de gradient",
              "Loi des grands nombres et théorème central limite", "Moyennes conditionnelles"]
    c.append(md("## 1. Programmes de la section 6 du polycopié"))
    for k, ti in zip(["p1","p2","p3","p4"], titres):
        c.append(md(f"### {ti}")); c.append(co(S[k]))
    c.append(md("## 2. Exercices\n### Exercice 1 — Matrices\nCalculer $A^\\top A$ et son inverse pour $A=\\begin{pmatrix}1&0\\\\1&1\\\\1&2\\end{pmatrix}$, et comparer avec votre calcul à la main (exercice 1 du polycopié)."))
    c.append(co("""A = np.array([[1, 0], [1, 1], [1, 2]])
AtA = A.T @ A
print(AtA)                       # [[3 3] [3 5]]
print(np.linalg.inv(AtA) * 6)    # 6 × inverse = [[5 -3] [-3 3]]""" if corrige else """A = np.array([[1, 0], [1, 1], [1, 2]])
# À compléter : A.T @ A, puis np.linalg.inv"""))
    c.append(md("### Exercice 2 — Formule de Bayes par simulation\nPrévalence 1 %, test positif chez 95 % des malades et négatif chez 90 % des non-malades. Simuler un million de personnes et estimer $\\mathbb P(M\\mid +)$ ; comparer à $0{,}0095/0{,}1085\\approx0{,}088$."))
    c.append(co("""rng = np.random.default_rng(11)
N = 1_000_000
malade = rng.random(N) < 0.01
positif = np.where(malade, rng.random(N) < 0.95, rng.random(N) < 0.10)
print(malade[positif].mean(), 0.0095 / 0.1085)""" if corrige else """rng = np.random.default_rng(11)
N = 1_000_000
malade = rng.random(N) < 0.01
# À compléter : simuler le résultat du test, puis la proportion de malades parmi les positifs"""))
    c.append(md("### Exercice 3 — Choix du pas\nPour $L(\\theta)=2\\theta^2$ (dérivée $4\\theta$), faire 50 pas de descente de gradient depuis $\\theta_0=1$ avec $\\eta=0{,}4$, puis $\\eta=0{,}6$. Comparer avec l'exercice 6 du polycopié."))
    c.append(co("""for eta in [0.4, 0.6]:
    theta = 1.0
    for _ in range(50):
        theta = theta - eta * 4 * theta
    print(f"eta = {eta} : theta_50 = {theta:.3e}")   # converge si eta < 0,5""" if corrige else """# À compléter : boucle de 50 pas pour eta = 0.4 puis 0.6"""))
    c.append(md("### Exercice 4 — Un biais qui paie\nPour $\\mu=1$, $\\sigma=3$, $n=10$, estimer par simulation l'erreur quadratique moyenne de $c\\,\\bar X_n$ pour $c\\in[0{,}5;1{,}2]$, puis comparer le minimum à $c^*=\\mu^2/(\\mu^2+\\sigma^2/n)$."))
    c.append(co("""rng = np.random.default_rng(12)
mu, sigma, n, R = 1.0, 3.0, 10, 200_000
xbar = rng.normal(mu, sigma, size=(R, n)).mean(axis=1)
cs = np.linspace(0.5, 1.2, 15)
eqm = np.array([np.mean((c * xbar - mu) ** 2) for c in cs])
print("c optimal simulé :", cs[eqm.argmin()].round(3), "| c* théorique :", round(mu**2 / (mu**2 + sigma**2 / n), 3))
print("EQM(c*) ≈", eqm.min().round(3), "contre EQM(1) ≈", np.mean((xbar - mu) ** 2).round(3))""" if corrige else """rng = np.random.default_rng(12)
mu, sigma, n, R = 1.0, 3.0, 10, 200_000
xbar = rng.normal(mu, sigma, size=(R, n)).mean(axis=1)
cs = np.linspace(0.5, 1.2, 15)
# À compléter : erreur quadratique moyenne de c * xbar pour chaque c, puis le minimum"""))
    nb = nbf.v4.new_notebook(); nb["cells"] = c
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb

out = Path(__file__).resolve().parent.parent / "notebooks"
nbf.write(build(False), out / "TP0.ipynb"); nbf.write(build(True), out / "corriges" / "TP0_corrige.ipynb")
print("ok")

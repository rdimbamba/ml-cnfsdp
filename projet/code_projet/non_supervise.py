# Volet non supervisé, commun à tous les groupes : à lancer après le code du groupe
# (il utilise Z, les variables actives, et illus : la cible et le poids).
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

cible = illus.columns[0]                     # variable illustrative : la cible
Zs = StandardScaler().fit_transform(Z)       # ACP normée : centrer, réduire

# 1. ACP : part d'inertie de chaque axe (éboulis), on garde les premiers axes
acp = PCA().fit(Zs)
print("Inertie expliquée (%) :", (100 * acp.explained_variance_ratio_).round(1))
q = 5                                  # nombre d'axes gardés, à choisir sur l'éboulis
coord = acp.transform(Zs)[:, :q]

# 2. Lire les axes : corrélation de chaque variable active avec les 3 premiers axes
cor = pd.DataFrame(acp.components_[:3].T * np.sqrt(acp.explained_variance_[:3]),
                   index=Z.columns, columns=["F1", "F2", "F3"])
display(cor.round(2).sort_values("F1"))

# 3. Premier plan factoriel, individus colorés selon la cible (illustrative)
plt.scatter(coord[:, 0], coord[:, 1], c=illus[cible], cmap="coolwarm", s=4, alpha=0.4)
plt.xlabel("F1"); plt.ylabel("F2"); plt.title(f"Premier plan, couleur : {cible}")
plt.show()

# 4. CAH de Ward sur les q premiers axes (sur 5 000 individus au plus, pour la mémoire)
rng = np.random.default_rng(0)
ech = rng.choice(len(coord), size=min(5000, len(coord)), replace=False)
arbre = linkage(coord[ech], method="ward")
dendrogram(arbre, truncate_mode="lastp", p=30); plt.title("Dendrogramme"); plt.show()
k = 4                                        # nombre de classes, lu sur le dendrogramme
classes_cah = fcluster(arbre, t=k, criterion="maxclust")

# 5. Consolidation : k-means partant des centres des classes de la CAH, sur tous
centres = np.vstack([coord[ech][classes_cah == c].mean(axis=0)
                     for c in range(1, k + 1)])
km = KMeans(n_clusters=k, init=centres, n_init=1).fit(coord)
classe = pd.Series(km.labels_ + 1, index=Z.index, name="classe")

# 6. Décrire les classes : effectif, moyennes des actives, taux pondéré de la cible
profil = Z.groupby(classe).mean().round(2)
profil.insert(0, "effectif", classe.value_counts().sort_index())
profil.insert(1, f"{cible} (%)", illus.groupby(classe).apply(
    lambda d: round(taux_pondere(d[cible], d["poids"]), 1)))
display(profil)

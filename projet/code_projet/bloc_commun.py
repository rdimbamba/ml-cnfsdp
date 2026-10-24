# Bloc commun à tous les groupes : à exécuter une fois, avant le code du groupe.
import os, re, glob
import numpy as np
import pandas as pd

DOSSIER = "donnees"   # dossier où les 4 archives .zip ont été décompressées

def chemin_dhs(fichier):
    """Chemin du fichier Stata du fichier 'HR', 'IR', 'KR' ou 'PR'."""
    motif = re.compile(rf"^CG{fichier}61\w*\.DTA$", re.IGNORECASE)
    trouves = [f for f in glob.glob(os.path.join(DOSSIER, "**", "*"), recursive=True)
               if motif.match(os.path.basename(f))]
    if not trouves:
        raise FileNotFoundError(f"Pas de CG{fichier}61...DTA dans « {DOSSIER} »")
    return trouves[0]

def lire_dhs(fichier, colonnes):
    """Lit les colonnes demandées, en gardant les codes numériques."""
    chemin = chemin_dhs(fichier)
    with pd.io.stata.StataReader(chemin) as lecteur:
        disponibles = set(lecteur.variable_labels())
    absentes = [c for c in colonnes if c not in disponibles]
    if absentes:
        print("Absentes du fichier, ignorées :", absentes)
    df = pd.read_stata(chemin, columns=[c for c in colonnes if c in disponibles],
                       convert_categoricals=False)
    print(f"{os.path.basename(chemin)} : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def manquant(serie, codes):
    """Remplace par NaN les codes spéciaux (ne sait pas, manquant, hors limites...)."""
    return serie.where(~serie.isin(codes))

def indicatrice(serie, codes, manquants=(99,)):
    """1 si le code est dans la liste, 0 sinon, NaN si le code est manquant."""
    x = serie.isin(codes).astype(float)
    return x.where(serie.notna() & ~serie.isin(manquants))

def taux_pondere(y, poids):
    """Proportion pondérée, en % : somme(poids * y) / somme(poids)."""
    return 100 * np.average(y, weights=poids)

def base_supervisee(df, cible, numeriques, categorielles):
    """X (une indicatrice par modalité des catégorielles), cible y, poids w."""
    num = [c for c in numeriques if c in df]
    cat = [c for c in categorielles if c in df]
    X = df[num + cat].copy()
    for c in cat:   # une modalité « manquant » plutôt que de perdre la ligne
        X[c] = X[c].astype("Int64").astype(str).replace("<NA>", "manquant")
    X = pd.get_dummies(X, columns=cat, dtype=int)
    return X, df[cible].astype(int), df["poids"]

def base_non_supervisee(df, actives, illustratives):
    """Variables actives (sans valeur manquante) et variables illustratives."""
    act = [c for c in actives if c in df]
    garde = df[act].notna().all(axis=1)
    print(f"Non supervisé : {garde.sum()} lignes complètes sur {len(df)}")
    return df.loc[garde, act], df.loc[garde, illustratives]

def resume(X, y, w):
    print("X :", X.shape, "| cible :", round(100 * y.mean(), 1), "% (non pondéré),",
          round(taux_pondere(y, w), 1), "% (pondéré)")
    manq = X.isna().sum()
    print("Valeurs manquantes à imputer :", dict(manq[manq > 0]))

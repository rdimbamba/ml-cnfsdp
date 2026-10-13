# Introduction au Machine Learning — CNFSDP 2026-2027

**Centre National de Formation en Statistique, Démographie et Planification** · Brazzaville, République du Congo
Licence 3 · Semestre 5 · UE Informatique IV · 13 heures · 1 crédit
Enseignant : **Rois Céti DIMBAMBA**, ingénieur statisticien économiste · actuaire · [CV en ligne](https://rdimbamba.shinyapps.io/roisceti_e-cv/)

Site du cours : <https://rdimbamba.github.io/ml-cnfsdp/>

Ce dépôt rassemble tous les supports du cours : diapositives, polycopié des outils mathématiques, TP en notebooks ouvrables dans Google Colab, et guide du projet de groupe. Le cours s'adresse à des débutants : chaque notion est posée par une définition mathématique simple, illustrée par un petit exemple chiffré, puis programmée en Python.

## Calendrier

| Séance | Date | Durée |
|---|---|---|
| Chapitre 0 · Outils mathématiques et installation (TP0) | à lire et faire chez soi avant le chapitre 1 | — |
| Chapitre 1 · Fondements | mercredi 14 octobre 2026 | 2 h |
| Chapitre 2 · Régression | vendredi 16 octobre 2026 | 2 h |
| Chapitre 3 · Classification et méthodes d'ensemble | annoncée en salle | 3 h |
| Chapitre 4 · Apprentissage non supervisé | annoncée en salle | 3 h |
| Chapitre 5 · Projet de bout en bout, interprétabilité et éthique | annoncée en salle | 3 h |
| Remise des rapports et soutenances du projet | fin novembre 2026 | — |

## Supports

| Chapitre | Diapositives | TP (Colab) |
|---|---|---|
| 0 · Les outils mathématiques du cours | [polycopié (PDF)](poly/chapitre0/chapitre0.pdf) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/TP0.ipynb) |
| 1 · Fondements du machine learning | [PDF](cours/chapitre1/chapitre1.pdf) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/TP1.ipynb) |
| 2 · Apprentissage supervisé : régression | [PDF](cours/chapitre2/chapitre2.pdf) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/TP2.ipynb) |
| 3 · Classification et méthodes d'ensemble | [PDF](cours/chapitre3/chapitre3.pdf) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/TP3.ipynb) |
| 4 · Apprentissage non supervisé | [PDF](cours/chapitre4/chapitre4.pdf) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/TP4.ipynb) |
| 5 · Projet de bout en bout, interprétabilité et éthique | [PDF](cours/chapitre5/chapitre5.pdf) | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rdimbamba/ml-cnfsdp/blob/main/notebooks/TP5.ipynb) |
| Projet de groupe | [Guide du projet (PDF)](projet/guide_projet.pdf) | — |

Les corrigés des TP sont publiés dans [`notebooks/corriges/`](notebooks/corriges/) après chaque séance.

## Démarrer

**Avant le chapitre 1**, lire le [polycopié du chapitre 0](poly/chapitre0/chapitre0.pdf) et faire le **TP0** (installation, une heure et demie environ).

**Option 1 : sur son ordinateur (recommandé).** Installer Miniconda, VS Code et Git (détails dans le polycopié, section TP0), puis :

```bash
git clone https://github.com/rdimbamba/ml-cnfsdp.git
cd ml-cnfsdp
conda env create -f environment.yml
conda activate ml-cnfsdp
code .
```

Pour récupérer les nouveaux supports au fil du semestre : `git pull` depuis le dossier `ml-cnfsdp`.

**Option 2 : sans rien installer.** Cliquer sur le bouton « Open in Colab » du TP voulu. Une copie du notebook s'ouvre dans le navigateur ; l'enregistrer dans son Google Drive (« Fichier », puis « Enregistrer une copie dans Drive ») pour garder son travail.

## Organisation du dépôt

```
ml-cnfsdp/
├── cours/chapitre1 … chapitre5   diapositives : PDF, sources LaTeX, figures, thème CNFSDP
├── poly/chapitre0                polycopié des outils mathématiques (PDF et sources)
├── notebooks/                    TP0 à TP5, versions à compléter
│   └── corriges/                 corrigés exécutés, publiés après chaque séance
├── projet/                       guide du projet de groupe, modèle de demande DHS
├── code/                         scripts des figures et des notebooks
├── docs/                         site du cours (GitHub Pages)
├── environment.yml               environnement conda du cours
└── requirements.txt              même liste pour pip
```

## Évaluation

| Épreuve | Poids |
|---|---|
| Examen sur table (3 h, sans document) | 40 % |
| Projet de groupe : rapport de 10 pages et notebook | 40 % |
| Projet de groupe : soutenance de 15 minutes | 20 % |

Les TP ne sont pas notés : ils se font en salle et se terminent chez soi si besoin.

## Projet et données d'enquête

Le projet porte sur l'**Enquête démographique et de santé du Congo 2011-2012** (programme DHS) : 9 groupes de 4 étudiants, une variante par groupe. Tout est décrit dans le [guide du projet](projet/guide_projet.pdf).

> **Les données DHS ne sont jamais déposées sur ce dépôt ni sur aucun site public.** Elles sont remises par l'enseignant dans le cadre d'une autorisation de classe, après signature de la charte d'utilisation. Le fichier `.gitignore` exclut le dossier `donnees/` et les fichiers Stata et SPSS : gardez-le dans vos propres dépôts. On rend le code et le rapport, jamais les données.

Les TP n'utilisent que des données fournies avec scikit-learn ou simulées dans le notebook : aucun téléchargement ni inscription n'est nécessaire.

## Questions

Posez vos questions dans l'onglet [Discussions](https://github.com/rdimbamba/ml-cnfsdp/discussions) du dépôt, pour que les réponses profitent à tous, ou par message privé à l'enseignant.

## Recompiler les supports

- Diapositives et polycopiés : `pdflatex` deux fois dans le dossier concerné (TeX Live, paquets `babel-french`, `beamer`, `tcolorbox`, `listings`).
- Figures : depuis `code/`, `python figures_chapitreN.py` (écrit dans `cours/chapitreN/`).
- Notebooks : depuis `code/`, `python make_tpN.py`, puis exécuter le corrigé avec `jupyter nbconvert --to notebook --execute --inplace ../notebooks/corriges/TPN_corrige.ipynb`.

---

© 2026 Rois Céti DIMBAMBA, CNFSDP. Supports du cours Introduction au Machine Learning.

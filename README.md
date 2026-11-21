<div align="center">

# 💼 EmployML — Classification & Analyse d'Employabilité des Candidats

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset Kaggle](https://img.shields.io/badge/Kaggle-70k+_Job_Applicants-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource)
[![Tests Passing](https://img.shields.io/badge/Tests-Passing-10B981?style=for-the-badge&logo=checkmarx&logoColor=white)](#-tests-unitaires-automatisés)
[![License: Academic](https://img.shields.io/badge/License-Academic-6366F1?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>Projet de Data Mining & Machine Learning Classique (Sans Deep Learning)</strong><br>
  Audit algorithmique, détection de règles déterministes cachées, analyse d'équité et simulateur didactique Streamlit.
</p>

</div>

---

## 📌 Présentation du Projet

Ce projet analyse le jeu de données Kaggle **[70k+ Job Applicants Data (Human Resource)](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource)** (73 462 candidats, 14 variables) dans l'objectif de classifier la situation d'emploi (`Employed`) d'un candidat à partir de son profil socioprofessionnel et technique.

Le pipeline complet (**EDA → Nettoyage → Prétraitement scikit-learn hermétique → Modélisation → Évaluation multi-angles → Déploiement**) aboutit à une application web interactive haute fidélité propulsée par **Streamlit**.

---

## 🔍 Découverte Majeure : Règle Déterministe Cachée

Une investigation statistique approfondie (notebook `03_classification_models.ipynb`, §6) démontre que la cible `Employed` est **une fonction logique exacte de la liste de technologies déclarées (`HaveWorkedWith`)** :

$$\text{Employed} = 1 \iff \sum_{i=1}^{10} \mathbb{I}(\text{tech}_i \in \text{HaveWorkedWith}) \ge 3$$

Où les 10 technologies clés sont :
> **Microsoft SQL Server**, **npm**, **jQuery**, **TypeScript**, **MongoDB**, **SQLite**, **Bash/Shell**, **Node.js**, **C#**, **Java**.

* **Conséquence directe** : Sur les 73 458 lignes valides, cette règle s'applique à **100.00 %**. Les modèles voyant cette variable atteignent donc des scores quasi-parfaits sans surapprentissage : ils ont simplement retrouvé la règle arithmétique sous-jacente.
* **Pertinence métier** : Cette cible ne reflète pas une employabilité réelle sur le marché du travail.
* **Expérience miroir (Profil seul)** : Dès que l'on retire la liste de technologies, tous les algorithmes s'effondrent à **≈ 57 % de F1-macro** (proche de la baseline aléatoire biaisée **Zero-R à 53.6 %**), démontrant que diplômes, années d'expérience et pays ne portent presque aucun signal.

---

## 🗂️ Structure du Dépôt

```
DM_Project/
├── data/
│   ├── raw/                 # stackoverflow_full.csv (Kaggle)
│   └── processed/           # applicants_clean.csv, train.csv, test.csv
├── models/                  # best_model.pkl, preprocessor.pkl, meta.json, métriques CSV
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb          # Nettoyage, imputations, tests Mann-Whitney & Cramér's V
│   ├── 02_preprocessing_encoding.ipynb   # Split stratifié 80/20, ColumnTransformer
│   └── 03_classification_models.ipynb    # GridSearchCV 3 plis, ablations, équité, règle
├── src/
│   ├── app.py               # Application Streamlit moderne (7 onglets thématiques)
│   ├── features.py          # ColumnTransformer scikit-learn partagé (anti data leakage)
│   ├── simulation.py        # Assembleur du simulateur HTML/Canvas
│   ├── simulation_v2.py     # Moteur de simulation didactique Streamlit
│   └── theme.py             # Design System & CSS (palette Glassmorphism Dark)
├── simulation/              # Bac à sable 2D autonome interactif (HTML5 Canvas / CSS / JS)
│   ├── css/style.css
│   ├── js/                  # Modules SVM, KNN, CART, Bayes
│   └── index.html
├── tests/
│   └── test_simulation.py   # Suite de tests unitaires automatisés
├── render.yaml              # Configuration de déploiement Cloud Render
├── requirements.txt         # Dépendances Python versionnées
└── .gitignore               # Configuration de filtrage stricte
```

---

## 📊 Benchmarking & Résultats

Tous les modèles ont été entraînés sous validation croisée stratifiée à 3 plis (`GridSearchCV`) en intégrant le `ColumnTransformer` au sein de chaque `Pipeline` pour garantir l'absence totale de fuite de données (*Data Leakage*).

### 1. Scénario Complet (Toutes variables avec technologies)

| Modèle | Accuracy | F1 macro | ROC-AUC | Statut |
|---|:---:|:---:|:---:|:---:|
| **Linear SVC** | **1.0000** | **1.0000** | **1.0000** | 🏆 **Champion** |
| Decision Tree (CART) | 0.9990 | 0.9990 | 0.9994 | Finaliste |
| Random Forest | 0.9742 | 0.9740 | 0.9977 | Ensembliste |
| K-Nearest Neighbors | 0.8716 | 0.8715 | 0.9655 | Métrique |
| Naive Bayes Multinomial | 0.8543 | 0.8529 | 0.9404 | Probabiliste |
| Zero-R (Baseline majoritaire) | 0.5362 | 0.3490 | *N/A* | Seuil d'utilité |

### 2. Scénario Profil Seul (Sans technologies ni `ComputerSkills`)

| Modèle | Accuracy | F1 macro | ROC-AUC |
|---|:---:|:---:|:---:|
| K-Nearest Neighbors | 0.5660 | 0.5487 | 0.5776 |
| Random Forest | 0.5813 | 0.5444 | 0.6020 |
| Naive Bayes Multinomial | 0.5743 | 0.5428 | 0.5916 |
| Linear SVC | 0.5744 | 0.5333 | 0.5929 |
| Decision Tree | 0.5709 | 0.5250 | 0.5678 |
| Zero-R (Baseline) | 0.5362 | 0.3490 | *N/A* |

---

## 🧪 Laboratoire de Simulation & XAI Multi-Modes

L'application intègre un banc d'essai didactique étendu (`src/simulation_v2.py`) articulé autour de **3 modes de simulation complémentaires** :

### Mode 1 : Décomposition Analytique des Algorithmes & Bac à Sable 2D Canvas
* **Linear SVC** : Visualisation de l'hyperplan optimal $\mathbf{w}^T \mathbf{x} + b = 0$, calcul de la marge géométrique $M = \frac{2}{\|\mathbf{w}\|}$ et inspection des coefficients d'importance.
* **KNN** : Sphéroïde des $K$ plus proches voisins, vote majoritaire vs vote pondéré ($1/d, 1/d^2$).
* **Decision Tree (CART)** : Partitionnement orthogonal 2D et cheminement décisionnel pas-à-pas (règles IF-THEN).
* **Naive Bayes** : Décomposition des vraisemblances et probabilités a posteriori.
* **Random Forest** : Décomposition du vote individuel de chaque arbre par ensachage (Bagging).

### Mode 2 : Analyse Contrefactuelle & Sensibilité What-If (Explainable AI)
* **Exploration Contrefactuelle (Recourse)** : Perturbez en direct les compétences et l'expérience du candidat pour observer les variations de distance à la frontière de décision ($w^T x + b$).
* **Détection Automatique du Point de Bascule (Flip 0 ➔ 1)** : Identification du chemin le plus court vers l'embauche (ajout minimal de technologies clés) et preuve de la stabilité de la règle déterministe.

### Mode 3 : Seuil Décisionnel & Matrice de Coûts Économiques (Cost-Sensitive Learning)
* **Tuning Dynamique du Seuil $\tau$** : Ajustement du seuil de coupure de probabilité $\tau \in [0.05, 0.95]$ avec recalcul instantané de la matrice de confusion $2 \times 2$.
* **Modélisation Financière des Erreurs** : Paramétrage du coût d'un faux positif ($C_{FP}$) versus coût d'un faux négatif ($C_{FN}$) et calcul automatique du seuil optimal $\tau^*$ minimisant la perte économique globale.

---

## ⚙️ Installation & Lancement Local

### Prérequis
* Python 3.11 ou supérieur
* Git

### Vérifications avant lancement

L'intégration continue du projet s'exécute avec **Python 3.11**. Pour
reproduire l'environnement évalué, privilégiez cette version lors de la
création de l'environnement virtuel.

L'interface nécessite les artefacts déjà versionnés dans `models/` :
`best_model.pkl`, `preprocessor.pkl`, `meta.json` et les fichiers CSV de
résultats. Si l'un d'eux est absent, relancez les notebooks dans l'ordre
`01` → `02` → `03` avant de démarrer Streamlit.

Les données source volumineuses ne sont pas versionnées. Certaines activités
du laboratoire de simulation peuvent donc utiliser un échantillon pédagogique
de remplacement lorsque `data/processed/` et `data/raw/stackoverflow_full.csv`
sont indisponibles. Ce remplacement ne constitue ni un nouveau résultat du
projet ni une évaluation supplémentaire du modèle.

```bash
# 1. Cloner le dépôt
git clone https://github.com/cris7suii/DM_Project.git
cd DM_Project

# 2. Créer et activer l'environnement virtuel
python -m venv venv
# Sous Windows :
.\venv\Scripts\activate
# Sous Linux / macOS :
source venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer l'application Streamlit
streamlit run src/app.py
```

### 🧪 Tests Unitaires Automatisés

Une suite de tests vérifie l'intégrité des modèles sérialisés, la règle déterministe et l'absence de régression :

```bash
python tests/test_simulation.py
```

---

## ⚖️ Responsabilité Éthique & Limites

1. **Reproduction d'Inégalités Structurelles** : Le jeu de données présente un écart de taux d'emploi historique entre sous-groupes (ex. 44.9 % chez les femmes contre 54.1 % chez les hommes). Le modèle reproduisant fidèlement les corrélations de données, il ne doit en aucun cas être utilisé comme outil décisionnel en recrutement.
2. **Artefact Synthétique** : L'existence de la règle déterministe sur les technologies clés illustre l'importance critique de l'audit des données (EDA) avant toute mise en production d'un système d'IA.
3. **Usage Académique** : Projet réalisé dans le cadre d'un module d'étude de Data Mining / Data Science.

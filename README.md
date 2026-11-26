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
├── models/                  # best_model.pkl, preprocessor.pkl, meta.json, tuning_results.csv, best_params.json, métriques CSV
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb          # Nettoyage, imputations, tests Mann-Whitney & Cramér's V
│   ├── 02_preprocessing_encoding.ipynb   # Split stratifié 80/20, ColumnTransformer
│   └── 03_classification_models.ipynb    # GridSearchCV 3 plis (dont AdaBoost, XGBoost), ablations, équité, règle
├── src/
│   ├── app.py               # Application Streamlit moderne (9 onglets thématiques, dont Tuning)
│   ├── cli.py               # Interface CLI d'évaluation unitaire/lot et audit du pipeline
│   ├── features.py          # Pipeline renforcé d'encodage et d'imputation (anti data leakage)
│   ├── simulation.py        # Assembleur du simulateur HTML/Canvas
│   ├── simulation_v2.py     # Moteur de simulation didactique Streamlit
│   └── theme.py             # Design System & CSS (palette Glassmorphism Dark)
├── simulation/              # Bac à sable 2D autonome interactif (HTML5 Canvas / CSS / JS)
│   ├── css/style.css
│   ├── js/                  # Modules SVM, KNN, CART, Bayes, AdaBoost
│   └── index.html
├── tests/
│   ├── test_simulation.py   # Suite de tests unitaires de la simulation
│   ├── test_boosting.py     # Tests des onglets AdaBoost / XGBoost et du mini-réglage GridSearchCV
│   └── test_preprocessor.py # Tests unitaires du pipeline renforcé d'encodage/imputation
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

<!-- À compléter après exécution du notebook 03 : lignes AdaBoost, XGBoost, Gradient Boosting et Régression logistique dans les deux tableaux de résultats. -->

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
* **AdaBoost** : Boosting de souches de décision — règle, erreur $\varepsilon_t$ et poids $\alpha_t$ de chaque tour, vote de chaque souche sur le candidat, courbe d'apprentissage train/test. Réglables : nombre de tours $T$, taux d'apprentissage $\eta$, profondeur des souches.
* **XGBoost** : Gradient boosting régularisé — évolution de $P(\text{Employé})$ arbre après arbre, courbe train/test, importance des variables (gain). Réglables : nombre d'arbres, $\eta$, profondeur, `subsample`, $\lambda$ (L2).
* **Réglage en direct** : les onglets AdaBoost et XGBoost embarquent un mini `GridSearchCV` (3 plis, F1-macro) rejouable depuis l'interface ; le réglage complet est dans le notebook 03 et consultable dans l'onglet **🎛️ Tuning**.

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

Le projet intègre une suite professionnelle et exhaustive de **123 tests automatisés** propulsée par **pytest** et **pytest-cov** pour certifier l'intégrité du pipeline Machine Learning, la conformité des prédictions, la détection des fuites de données, la robustesse du préprocesseur renforcé, les scénarios contrefactuels XAI et la stabilité fonctionnelle de l'application.

#### Installation des dépendances de test
Les dépendances de test (`pytest>=8.0`, `pytest-cov>=5.0`) sont incluses dans `requirements.txt` :
```bash
pip install -r requirements.txt
```

#### Exécution de la suite de tests
```bash
# Exécution verbeuse de tous les tests avec pytest
python -m pytest -v

# Exécution avec rapport de couverture de code détaillé
python -m pytest -v --cov=src --cov-report=term-missing

# Exécution autonome des tests du préprocesseur renforcé
python tests/test_preprocessor.py

# Exécution autonome des tests de la simulation
python tests/test_simulation.py

# Audit comparatif du pipeline en ligne de commande (Baseline vs Renforcé)
python src/cli.py --audit-pipeline
```

#### Organisation des modules de test
* **`tests/test_artifacts.py`** : Validation de la présence physique, de la désérialisation, de la structure des métadonnées `meta.json`, de la détection de modèles corrompus ou invalides et de l'intégrité des tableaux CSV (`best_model.pkl`, `preprocessor.pkl`, `results.csv`, `fairness.csv`, etc.).
* **`tests/test_prediction.py`** : Validation des dimensions et types des prédictions, conformité mathématique de la marge Linear SVC, respect strict de la règle déterministe ($\ge 3$ technologies clés), invariance sociodémographique, gestion des arguments et modes batch de `src/cli.py`.
* **`tests/test_candidate.py`** : Validation des contrôles de cohérence sur les profils candidats (`validate_candidate`), extraction des technologies clés et calcul des synthèses de règles.
* **`tests/test_preprocessor.py`** : Validation unitaire approfondie du **pipeline renforcé d'encodage et d'imputation** (`build_reinforced_preprocessor`), encodage ordinal (`EdLevel`, `Age`), binaire compact (`drop='if_binary'`), capture MNAR (`MissingIndicator`), mise à l'échelle robuste (`RobustScaler`), sanctuarisation des 10 technologies clés (`KEY_TECHS`) et résilience aux NaNs et catégories inconnues.
* **`tests/test_preprocessing.py`** : Validation unitaire de `features.py` (`split_techs`, `_log1p`, `build_preprocessor`), gestion des valeurs manquantes (`SimpleImputer`), robustesse aux catégories inconnues (`OneHotEncoder`) et vérification de la non-mutation des DataFrames d'entrée.
* **`tests/test_boosting.py`** : Validation de l'intégration des algorithmes de boosting (AdaBoost, XGBoost conditionnel) et conformité des interfaces d'entraînement et d'inférence.
* **`tests/test_simulation.py`** : Laboratoire didactique (`simulation_v2.py`), reproductibilité des tirages synthétiques (`seed`), inférence des 6 architectures, analyse contrefactuelle What-If (invariance face aux variables neutres, impact sélectif des technologies clés vs non-clés, presets de profils), optimisation de la matrice de coûts économiques ($FP \cdot C_{FP} + FN \cdot C_{FN}$) et calcul du seuil optimal $\tau^*$.
* **`tests/test_app_flow.py`** : Exécution fonctionnelle complète d'application (`src/app.py`), gestion des erreurs de démarrage et d'artefacts, interception des incohérences de saisie de profil (`st.stop`) et génération/exportation du rapport de gouvernance JSON.
* **`tests/test_app_logic.py`** : Validation unitaire de l'assembleur HTML (`simulation.build_html`) et des composants de rendu du design system (`theme.py` : badges, cartes hero, KPIs, bannières éthiques, alertes) isolés du serveur Streamlit.

#### Matrice de Couverture de Code (`pytest-cov`)

| Module | Rôle & Composants | Couverture | Statut |
|---|---|:---:|:---:|
| `src/artifacts.py` | Gestion et chargement sécurisé des artefacts sérialisés | **100 %** | ✅ Intégral |
| `src/candidate.py` | Règles métier candidat & formatage de marge | **100 %** | ✅ Intégral |
| `src/simulation.py` | Assembleur HTML du simulateur Canvas | **100 %** | ✅ Intégral |
| `src/theme.py` | Design system & injection CSS | **100 %** | ✅ Intégral |
| `src/app.py` | Application interactive Streamlit & export d'audit | **92 %** | ✅ Haute couverture |
| `src/cli.py` | Interface ligne de commande, inférence batch & audit | **91 %** | ✅ Haute couverture |
| `src/simulation_v2.py` | Vues 2D, analyse What-If & matrice de coûts $\tau^*$ | **80 %** | ✅ Haute couverture |
| `src/features.py` | Encodage renforcé, imputation MNAR & scalers | **78 %** | ✅ Haute couverture |
| **Ensemble du projet** | **123 tests automatisés unitaires et d'intégration** | **83 %** | 🏆 **Certifié** |

---

## ⚖️ Responsabilité Éthique & Limites

1. **Reproduction d'Inégalités Structurelles** : Le jeu de données présente un écart de taux d'emploi historique entre sous-groupes (ex. 44.9 % chez les femmes contre 54.1 % chez les hommes). Le modèle reproduisant fidèlement les corrélations de données, il ne doit en aucun cas être utilisé comme outil décisionnel en recrutement.
2. **Artefact Synthétique** : L'existence de la règle déterministe sur les technologies clés illustre l'importance critique de l'audit des données (EDA) avant toute mise en production d'un système d'IA.
3. **Usage Académique** : Projet réalisé dans le cadre d'un module d'étude de Data Mining / Data Science.
---

## 📁 Contribution documentaire

Dans le cadre du mini-projet **EmployML**, cette contribution regroupe les livrables documentaires associés au projet.

### Documents fournis

- **Rapport du mini-projet :** [`Rapport_MiniProject_RSD23.pdf`](docs/Rapport_MiniProject_RSD23.pdf)
- **Index des captures :** [`CAPTURES.md`](docs/CAPTURES.md)

### Captures de l'application

Les captures suivantes, extraites du rapport scientifique, illustrent différentes fonctionnalités de l'application EmployML :

| Capture | Description |
|---|---|
| `page_14_capture_01.png` | Accueil et interface de l'application |
| `page_14_capture_02.png` | Inférence et prédiction |
| `page_15_capture_01.png` | Benchmarking |
| `page_15_capture_02.png` | Analyse d'ablation |
| `page_16_capture_01.png` | Équité et IA responsable |



---

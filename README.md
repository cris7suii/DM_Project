# 💼 Classification des candidats selon leur employabilité

Projet de **Data Mining / Machine Learning classique** (sans Deep Learning) : prédire si un candidat est en situation d'emploi (`Employed`) à partir de son profil, avec le jeu de données Kaggle [**70k+ Job Applicants Data (Human Resource)**](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource) (73 462 candidats, 14 variables).

Le pipeline complet (nettoyage → prétraitement → modèles → évaluation) aboutit à une application **Streamlit** de démonstration.

## ⚠️ Résultat principal : la cible est une règle déterministe

La cible `Employed` est **une fonction exacte de la liste de technologies `HaveWorkedWith`** :

> `Employed = 1` ⇔ le candidat maîtrise **au moins 3** des 10 technologies suivantes : Microsoft SQL Server, npm, jQuery, TypeScript, MongoDB, SQLite, Bash/Shell, Node.js, C#, Java.

Cette règle est exacte sur les 73 458 lignes (notebook 03, §6). Les modèles qui voient cette liste atteignent donc ≈ 100 %, sans fuite de données : ils retrouvent la règle. **Ce n'est pas une mesure de l'employabilité réelle.** Sans la liste de technologies, tous les modèles restent proches de la baseline (≈ 57 % contre 53,6 %).

## 🗂️ Structure

```
DM_Project/
├── data/raw/            # placer stackoverflow_full.csv ici (Kaggle)
├── data/processed/      # applicants_clean.csv, train.csv, test.csv (générés)
├── models/              # best_model.pkl, results.csv, ablation.csv, fairness.csv, ... (générés)
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_preprocessing_encoding.ipynb
│   └── 03_classification_models.ipynb
├── src/
│   ├── features.py      # préprocesseur partagé (ColumnTransformer)
│   └── app.py           # application Streamlit
├── .streamlit/config.toml
└── requirements.txt
```

## 📊 Démarche

1. **EDA et nettoyage** : types, valeurs manquantes (63 `HaveWorkedWith`), doublons (4), outliers (IQR, conservés), cohérence (588 lignes avec `YearsCodePro > YearsCode`, corrigées), corrélations, tests **Mann-Whitney U** et **χ² + V de Cramér**.
2. **Prétraitement** : split stratifié 80/20 (`random_state=42`) ; numériques (imputation médiane + standardisation, `log1p` sur le salaire) ; catégorielles (imputation + One-Hot, pays rares regroupés) ; `HaveWorkedWith` en **multi-hot** des 40 technologies les plus fréquentes.
3. **Modèles** : Zero-R, Multinomial Naive Bayes, KNN, Arbre de décision, Random Forest, Linear SVC, réglés par `GridSearchCV` (3 plis stratifiés) sur des `Pipeline` (le préprocesseur est ré-ajusté à chaque pli : pas de fuite). Le champion est choisi sur le **F1-macro en validation croisée**, pas sur le test. Le réglage se fait sur un sous-échantillon de 20 000 lignes (`FAST_MODE`), puis les meilleurs hyperparamètres sont ré-entraînés sur tout le train.
4. **Ablations, diagnostic de la règle, analyse d'équité** (genre, âge, santé mentale, accessibilité).

### Résultats (jeu de test, toutes variables)

| Modèle | Accuracy | F1 macro | ROC-AUC |
|---|---|---|---|
| **Linear SVC (champion)** | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 0.9990 | 0.9990 | 0.9994 |
| Random Forest | 0.9742 | 0.9740 | 0.9977 |
| K-Nearest Neighbors | 0.8716 | 0.8715 | 0.9655 |
| Multinomial Naive Bayes | 0.8543 | 0.8529 | 0.9404 |
| Zero-R (baseline) | 0.5362 | 0.3490 | N/A |

### Résultats sans technologies ni `ComputerSkills` (profil seul)

| Modèle | Accuracy | F1 macro | ROC-AUC |
|---|---|---|---|
| K-Nearest Neighbors | 0.5660 | 0.5487 | 0.5776 |
| Random Forest | 0.5813 | 0.5444 | 0.6020 |
| Multinomial Naive Bayes | 0.5743 | 0.5428 | 0.5916 |
| Linear SVC | 0.5744 | 0.5333 | 0.5929 |
| Decision Tree | 0.5709 | 0.5250 | 0.5678 |
| Zero-R (baseline) | 0.5362 | 0.3490 | N/A |

## ⚙️ Installation et exécution

```bash
git clone https://github.com/cris7suii/DM_Project.git
cd DM_Project
python -m venv venv && source venv/bin/activate     # Windows : venv\Scripts\activate
pip install -r requirements.txt
# télécharger le jeu de données sur Kaggle (lien en haut de ce README), placer stackoverflow_full.csv dans data/raw/,
# puis exécuter les notebooks 01 -> 02 -> 03
streamlit run src/app.py
```

## ⚠️ Limites

- La cible est construite à partir des technologies déclarées : elle ne mesure pas l'employabilité réelle.
- `Employment` n'est pas documenté dans le jeu de données et n'a presque aucun effet.
- Le jeu contient des attributs sensibles (genre, âge, santé mentale, accessibilité) : ne jamais utiliser un tel modèle pour décider d'un recrutement. L'écart de taux d'emploi entre groupes (ex. femmes 44,9 % vs hommes 54,1 %) est déjà présent dans l'étiquette.
- Projet académique ; les prédictions sont indicatives.

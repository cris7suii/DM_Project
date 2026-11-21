# 🎓 Rapport Méthodologique & Guide de Préparation à la Soutenance

**Module** : Data Mining & Machine Learning Classique (Sans Deep Learning)  
**Auteur** : `l3lmiouchi-prog`  
**Projet** : EmployML — Classification & Analyse d'Employabilité (Kaggle 70k+ Applicants)  

---

## 📑 Sommaire
1. [Problématique & Données](#1-problématique--données)
2. [Protocole Anti-Fuite (No Data Leakage)](#2-protocole-anti-fuite-no-data-leakage)
3. [Pourquoi le F1-Macro et non l'Accuracy ?](#3-pourquoi-le-f1-macro-et-non-laccuracy-)
4. [Formulation Mathématique du Champion (Linear SVC)](#4-formulation-mathématique-du-champion-linear-svc)
5. [Démonstration de la Règle Déterministe Cachée](#5-démonstration-de-la-règle-déterministe-cachée)
6. [Audit d'Équité & IA Responsable](#6-audit-déquité--ia-responsable)
7. [Fiches Questions / Réponses pour la Soutenance Orale](#7-fiches-questions--réponses-pour-la-soutenance-orale)

---

## 1. Problématique & Données

Le jeu de données Kaggle **70k+ Job Applicants Data** comporte 73 462 observations et 14 variables hétérogènes (numériques, catégorielles nominales, chaînes semi-structurées de technologies `HaveWorkedWith`, attributs sensibles).

L'objectif est de prédire la variable binaire `Employed` ($\in \{0, 1\}$) tout en respectant une contrainte forte : **aucun modèle de Deep Learning**.

---

## 2. Protocole Anti-Fuite (No Data Leakage)

L'une des erreurs les plus sévèrement pénalisées en Data Mining est la **fuite d'information (*Data Leakage*)** entre le jeu d'entraînement et le jeu de test.

### Notre Architecture Hermétique :
1. **Partitionnement Préalable** : Un split stratifié 80/20 avec graine fixe (`random_state=42`) isole immédiatement $X_{\text{test}}$ et $y_{\text{test}}$.
2. **Pipelines Scikit-Learn Encapsulés** :
   * L'imputation médiane des variables numériques, le calcul de la moyenne/écart-type (`StandardScaler`), l'apprentissage du vocabulaire des 40 technologies (`CountVectorizer`) et l'encodage One-Hot (`OneHotEncoder`) sont intégrés dans un `ColumnTransformer`.
   * Lors de la validation croisée `GridSearchCV` (3 plis stratifiés), le `ColumnTransformer` est **ré-ajusté uniquement sur les 2 plis d'entraînement** et simplement projeté sur le pli de validation.
   * **Conséquence** : Aucune statistique globale du test set ne contamine les modèles.

---

## 3. Pourquoi le F1-Macro et non l'Accuracy ?

Dans notre jeu de données, la classe majoritaire représente **53.6 %** des candidats.

* **Piège de l'Accuracy** : Un modèle trivial (*Zero-R*) qui prédit toujours la classe majoritaire obtient une accuracy de **53.62 %** tout en étant totalement inutile en pratique.
* **Choix du F1-Macro** : Le F1-macro accorde un poids identique à chaque classe sans favoriser la classe sur-représentée :
  $$\text{F1-macro} = \frac{\text{F1}_0 + \text{F1}_1}{2}$$
  Pour Zero-R, le F1 sur la classe minoritaire est nul ($\text{F1}_0 = 0$), ce qui donne $\text{F1-macro} = 0.3490$. Cette métrique pénalise lourdement les modèles paresseux et reflète la véritable capacité de discrimination.

---

## 4. Formulation Mathématique du Champion (Linear SVC)

Le modèle sélectionné comme champion est le **Linear Support Vector Classifier (Linear SVC)** avec fonction de perte *Squared Hinge* ($L2$) et régularisation $C$.

### Problème d'Optimisation Primal (Soft Margin) :
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b))^2$$

* $\mathbf{w}$ : vecteur normal de l'hyperplan séparateur.
* $b$ : biais (ordonnée à l'origine).
* $M = \frac{2}{\|\mathbf{w}\|}$ : largeur de la marge géométrique séparant les deux classes.
* $C$ : paramètre de compromis entre maximisation de la marge et pénalisation des violations.

### Règle d'Inférence :
$$\hat{y}(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$

La distance algébrique $d(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ sert d'indicateur direct de confiance dans notre interface Streamlit et notre CLI.

---

## 5. Démonstration de la Règle Déterministe Cachée

Lors de l'analyse exploratoire (notebook 03, §6), une anomalie majeure a été mise au jour :

$$\text{Employed} = 1 \iff \sum_{k \in \mathcal{K}} \mathbb{I}(k \in \text{HaveWorkedWith}) \ge 3$$

Avec $\mathcal{K} = \{$ Microsoft SQL Server, npm, jQuery, TypeScript, MongoDB, SQLite, Bash/Shell, Node.js, C#, Java $\}$.

### Tableau de Contingence Empirique sur 73 458 Candidats :
* Si $n_{\text{clés}} \ge 3$ : **100.00 %** sont `Employed = 1` (0 exception).
* Si $n_{\text{clés}} < 3$ : **100.00 %** sont `Employed = 0` (0 exception).

**Conclusion Scientifique** : Les performances parfaites ($F_1 = 1.0000$) ne résultent pas d'une capacité miraculeuse à prédire l'employabilité humaine, mais de la redécouverte arithmétique de la fonction ayant généré la colonne cible synthétique de Kaggle. Sans cette colonne, tous les modèles retombent à $\approx 57\%$ de F1-macro.

---

## 6. Audit d'Équité & IA Responsable

L'évaluation des disparités algorithmiques révèle des écarts statistiques significatifs :
* **Genre** : Taux d'emploi prédit de 54.1 % chez les hommes contre 44.9 % chez les femmes.
* **Cause** : Cet écart découle directement de la sous-représentation historique des femmes dans la liste des 10 technologies clés au sein du questionnaire.
* **Préconisation Déontologique** : L'utilisation directe de ce modèle à des fins de tri automatique de CVs constituerait une discrimination systémique illégale et éthiquement inacceptable.

---

## 7. Fiches Questions / Réponses pour la Soutenance Orale

### Q1 : « Pourquoi n'avez-vous pas utilisé de Deep Learning (ex. Réseau de neurones, MLP) ? »
> **Réponse** : Le cahier des charges académique imposait le Machine Learning classique. De plus, sur des données tabulaires hétérogènes de taille moyenne (~70k lignes) régies par une frontière linéaire stricte, un Linear SVC ou un arbre de décision est plus rapide, plus interprétable et mathématiquement optimal par rapport à un réseau de neurones lourd.

### Q2 : « Pourquoi Random Forest (97.4%) est-il inférieur au Linear SVC (100%) ? »
> **Réponse** : Random Forest utilise un sous-échantillonnage aléatoire des variables (`max_features`) à chaque split. Quand les technologies clés ne sont pas sélectionnées dans un sous-arbre, celui-ci commet des erreurs de vote, ce qui réduit légèrement la performance globale par rapport à un hyperplan global qui pondère directement l'ensemble des 40 technologies.

### Q3 : « Quelle est l'utilité du mode What-If / XAI ajouté dans votre application ? »
> **Réponse** : Il répond aux exigences modernes d'Explainable AI (ex. RGPD / AI Act). Un candidat refusé a le droit de savoir quelle modification minimale de son profil (recourse contrefactuel) lui permettrait de franchir la frontière de décision.

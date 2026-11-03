# Cahier des charges — Simulation interactive des algorithmes de Data Mining

## 1. Contexte du projet

Le projet `DM_Project_complet` est un projet de Data Mining / Machine Learning classique qui prédit la classe `Employed` à partir du jeu de données **70k+ Job Applicants Data (Human Resource)**. Il utilise Python, pandas et scikit-learn, avec une interface Streamlit et un simulateur HTML/CSS/JavaScript.

La partie à développer est **la simulation pédagogique des algorithmes**. Le dossier `45` est un exemple de référence visuelle et pédagogique : il présente des simulations de KNN, arbre de décision, Naive Bayes et SVM. Il faut s'en inspirer pour la richesse des explications, les visualisations, les interactions et l'organisation, **sans copier son interface, son code ou sa structure à l'identique**.

## 2. Objectif fonctionnel

Créer une interface interactive qui permette de comprendre comment plusieurs algorithmes classent un candidat du jeu de données réel, comment leurs décisions diffèrent, et quels critères/mesures permettent de comparer leurs résultats.

L'utilisateur doit pouvoir :
- sélectionner un candidat réel depuis le jeu de données ou charger un échantillon réel ;
- voir ses caractéristiques utiles à la prédiction ;
- lancer ou afficher la prédiction de plusieurs algorithmes ;
- comprendre, visuellement et textuellement, pourquoi chaque algorithme aboutit à sa classe ;
- comparer les prédictions entre algorithmes et, si la vraie étiquette est disponible, les comparer à `Employed` ;
- consulter des mesures calculées sur un jeu de test séparé ;
- explorer l'effet de certains paramètres (par exemple K pour KNN, profondeur pour l'arbre, C pour SVM, lissage pour Naive Bayes) lorsque cela est pertinent ;
- distinguer clairement une démonstration locale sur un candidat des performances globales mesurées sur le jeu de test.

La simulation doit être fondée sur les **données et les variables du projet**, pas sur un jeu de points entièrement artificiel. Des points synthétiques peuvent éventuellement servir à expliquer une notion isolée, mais ils ne doivent pas remplacer les résultats issus du vrai dataset.

## 3. Compréhension du dataset et du modèle actuel

Le README et les notebooks du projet indiquent :
- Dataset : `70k+ Job Applicants Data (Human Resource)`, environ 73 462 lignes au départ.
- Cible : `Employed` (binaire : 1 ou 0).
- Variables importantes :
  - numériques : `YearsCode`, `YearsCodePro`, `PreviousSalary`, `ComputerSkills`, `Employment` ;
  - catégorielles : `Age`, `Accessibility`, `EdLevel`, `Gender`, `MentalHealth`, `MainBranch`, `Country` ;
  - technologies déclarées : `HaveWorkedWith`, une liste de technologies séparées par `;`.
- Prétraitement partagé : `src/features.py`, avec imputation, mise à l'échelle, One-Hot Encoding et multi-hot des technologies.
- Modèles déjà comparés dans `notebooks/03_classification_models.ipynb` : Zero-R (baseline), Multinomial Naive Bayes, KNN, Decision Tree, Random Forest et Linear SVC.
- Modèle champion actuellement enregistré dans `models/meta.json` : Linear SVC.
- Artefacts présents : `models/best_model.pkl`, `preprocessor.pkl`, `results.csv`, `profile_only_results.csv`, `ablation.csv`, `fairness.csv`, `diagnostic.csv` et `meta.json`.
- Interface existante : `src/app.py`, `src/simulation.py` et `simulation/` (HTML, CSS, JavaScript).

### Résultats existants à reprendre comme point de départ, pas comme résultats garantis d'une nouvelle simulation

Sur le jeu de test avec toutes les variables, `models/results.csv` rapporte :
- Linear SVC : Accuracy 1.0000, F1-macro 1.0000, ROC-AUC 1.0000 ;
- Decision Tree : Accuracy environ 0.9990, F1-macro environ 0.9990 ;
- Random Forest : Accuracy environ 0.9742, F1-macro environ 0.9740 ;
- KNN : Accuracy environ 0.8716, F1-macro environ 0.8715 ;
- Multinomial Naive Bayes : Accuracy environ 0.8543, F1-macro environ 0.8529 ;
- Zero-R : Accuracy environ 0.5362, F1-macro environ 0.3490.

Sans les technologies et `ComputerSkills`, les résultats sont proches de la baseline (environ 0.53–0.58 d'accuracy, selon le modèle).

**Avertissement méthodologique essentiel :** l'analyse du notebook 03 et le README concluent que `Employed` est presque une règle déterministe construite à partir de `HaveWorkedWith` : `Employed = 1` si au moins 3 des 10 technologies clés sont présentes. Les scores proches de 100 % ne démontrent donc pas que le système sait évaluer l'employabilité réelle. Il apprend essentiellement une règle intégrée au dataset. La simulation doit montrer cette limite, pas la masquer.

En conséquence :
1. Ne jamais présenter `Employed` comme une mesure fiable de la capacité réelle à recruter une personne.
2. Ne pas afficher « meilleur candidat à embaucher » comme recommandation automatique.
3. Utiliser une formulation comme « classe prédite par le modèle » ou « prédiction de la variable Employed ».
4. Inclure une note indiquant que l'outil est pédagogique et ne doit pas servir à prendre une décision réelle de recrutement.
5. Si les attributs `Gender`, `Age`, `MentalHealth` ou `Accessibility` sont visibles, expliquer les risques éthiques et éviter de les transformer en critères de sélection de personnes.

## 4. Algorithmes à simuler

Priorité recommandée : reprendre les algorithmes déjà évalués dans le projet pour que la simulation soit cohérente avec les résultats existants.

### A. K-Nearest Neighbors (KNN)
- Montrer le candidat choisi et ses voisins les plus proches dans un espace de visualisation explicable.
- Afficher le K réglable, les distances et les classes des voisins.
- Expliquer le vote majoritaire et, si possible, le vote pondéré par la distance.
- Montrer comment la prédiction varie quand K change.
- Les distances doivent être calculées sur les variables prétraitées, ou leur approximation visuelle doit être clairement signalée.

### B. Arbre de décision
- Montrer les principaux nœuds/règles qui conduisent à une classe.
- Afficher les variables de séparation, seuils, profondeur et importance des variables lorsque disponibles.
- Permettre de limiter la profondeur pour expliquer le compromis entre simplicité et surapprentissage.
- Utiliser l'arbre réellement entraîné sur les données du projet ; ne pas fabriquer des règles qui ne viennent pas du modèle.

### C. SVM linéaire (Linear SVC)
- Expliquer l'hyperplan de décision, la marge et le rôle de `C`.
- Montrer le score de décision du candidat (si disponible) et la classe prédite.
- Préciser que le score de décision n'est pas une probabilité.
- Si une frontière est projetée en 2D, indiquer qu'elle représente une projection ou une approximation et non l'espace complet des variables.

### D. Naive Bayes
- Utiliser la variante réellement choisie et entraînée dans le projet (actuellement Multinomial Naive Bayes dans les résultats).
- Afficher les probabilités de classe si `predict_proba` est disponible, ainsi que les probabilités a priori et les contributions explicables lorsque possible.
- Ne pas présenter une simulation de Naive Bayes gaussien en 2D comme s'il s'agissait du MultinomialNB entraîné dans le projet.
- Expliquer brièvement l'hypothèse d'indépendance conditionnelle et ses limites.

### E. Random Forest (recommandé)
- Le modèle figure déjà dans la comparaison du projet.
- Montrer le vote des arbres, la proportion de votes par classe et les variables importantes si le modèle est entraîné et accessible.
- S'il est difficile de visualiser chaque arbre, privilégier les votes et l'importance des variables.

### F. Baseline Zero-R (facultatif, mais utile)
- Afficher la classe majoritaire comme prédiction de référence.
- Expliquer pourquoi un modèle complexe doit être comparé à une baseline simple.

## 5. Données : règles non négociables

1. Inspecter d'abord les fichiers du projet et réutiliser les chemins, schémas et fonctions déjà présents.
2. Ne pas inventer de nouvelles colonnes ou de nouvelles valeurs de candidats.
3. Ne pas supposer que le fichier brut est présent : le ZIP du projet ne contient pas nécessairement le dataset original dans `data/raw/`. Vérifier l'existence des fichiers à l'exécution.
4. Si les données brutes sont absentes, afficher un message d'installation clair demandant de placer le fichier attendu (par exemple `data/raw/stackoverflow_full.csv`) ou d'utiliser les artefacts déjà disponibles si ceux-ci suffisent à la vue demandée. Ne pas remplacer silencieusement les données manquantes par des données aléatoires.
5. Utiliser le prétraitement du projet (`src/features.py`) ou le pipeline sauvegardé, afin de garder une cohérence avec l'entraînement.
6. Pour mesurer les performances, utiliser le split train/test existant ou reconstruire le split avec les paramètres du notebook (`test_size=0.2`, `stratify=y`, `random_state=42`) si le dataset nettoyé est disponible.
7. Éviter toute fuite de données : ajuster le prétraitement sur le train uniquement ; ne pas entraîner et évaluer sur les mêmes lignes.
8. Afficher les métriques réellement calculées à partir des prédictions, avec les noms des fichiers/sources et la taille de l'échantillon.
9. Si la simulation utilise un échantillon pour des raisons de performance, indiquer la taille, la méthode de sélection et la graine aléatoire.
10. Séparer clairement :
    - **prédiction individuelle** pour le candidat sélectionné ;
    - **performance globale** sur le jeu de test ;
    - **illustration 2D** destinée à l'explication.

## 6. Interface utilisateur souhaitée

Créer une interface propre, moderne et cohérente avec le projet actuel. Le thème visuel du dossier `45` peut inspirer le contraste, la hiérarchie et les panneaux d'explication, mais il ne faut pas le reproduire à l'identique.

Structure suggérée :
1. **Introduction** : but de la simulation, dataset utilisé, cible, avertissement sur la limite de la variable `Employed`.
2. **Sélection du candidat** : choix d'une ligne réelle, identifiant de ligne non sensible ou filtres utiles ; affichage des variables retenues.
3. **Choix de l'algorithme** : onglets/cartes KNN, arbre, SVM, Naive Bayes, Random Forest et baseline.
4. **Visualisation explicative** : voisinage, règles de l'arbre, score/marge SVM, votes de forêt ou probabilités de classe.
5. **Comparaison des prédictions** : tableau par algorithme avec classe prédite, vraie classe si connue, et accord/désaccord.
6. **Performance globale** : Accuracy, Precision, Recall, F1-macro, matrice de confusion et ROC-AUC uniquement si elle est calculable.
7. **Interprétation** : explication en langage simple de la décision et des limites.
8. **Éthique et limites** : cible artificielle, attributs sensibles, absence de décision de recrutement réelle.

Éléments interactifs possibles :
- choix d'un candidat réel ;
- réglage de K, de la profondeur de l'arbre et de C pour SVM ;
- sélection du mode « toutes les variables » versus « profil seul sans `HaveWorkedWith` ni `ComputerSkills` » ;
- comparaison des prédictions et des métriques entre les deux scénarios ;
- affichage de la vraie classe uniquement si elle est disponible, avec un libellé qui ne la confond pas avec une recommandation.

## 7. Architecture technique proposée

Respecter d'abord l'architecture existante :
- `src/app.py` : intégrer la simulation dans l'onglet Streamlit existant ou l'améliorer sans casser les autres onglets ;
- `src/features.py` : réutiliser le prétraitement existant ;
- `models/` : charger les artefacts existants si leur format correspond au besoin ;
- `simulation/` : organiser les ressources HTML/CSS/JS si une visualisation web autonome reste pertinente ;
- `src/simulation.py` : conserver ou adapter la génération HTML pour l'intégration Streamlit.

Ne pas créer une seconde application inutile si la simulation peut être intégrée à l'application actuelle. Éviter de dupliquer la logique de prétraitement ou de réécrire tout le projet.

Si une visualisation web en JavaScript ne peut pas facilement utiliser les prédictions du modèle Python, proposer une interface claire entre Streamlit/Python et la visualisation, ou réaliser les graphiques en Python. Ne pas faire croire qu'un algorithme JavaScript simplifié est le modèle scikit-learn réel.

## 8. Critères d'acceptation

La partie simulation est acceptable si :
- [ ] elle utilise les données réelles du projet ou signale explicitement leur absence ;
- [ ] elle utilise des modèles entraînés sur ces données et un prétraitement cohérent ;
- [ ] elle explique au moins KNN, arbre de décision, SVM linéaire et Naive Bayes ;
- [ ] elle inclut Random Forest si cela peut être fait sans complexité disproportionnée ;
- [ ] les paramètres interactifs influencent réellement le calcul correspondant, et pas seulement l'affichage ;
- [ ] les résultats individuels et les performances globales sont séparés ;
- [ ] les métriques sont calculées et non codées en dur ;
- [ ] elle montre clairement la différence entre le scénario complet et le scénario « profil seul » ;
- [ ] elle explique le problème de la règle intégrée dans la cible `Employed` ;
- [ ] elle ne formule aucune recommandation réelle d'embauche ;
- [ ] elle conserve les autres fonctions de l'application ;
- [ ] le code est commenté, structuré, testable et accompagné d'instructions de lancement ;
- [ ] les fichiers existants ne sont pas écrasés avant d'avoir été inspectés et les changements sont limités à ce qui est nécessaire.

## 9. Livrables attendus du LLM

Avant de coder :
1. Résumer la structure actuelle du projet.
2. Identifier les fichiers à modifier/créer et expliquer pourquoi.
3. Vérifier la disponibilité des données brutes et des modèles sauvegardés.
4. Décrire les algorithmes et les visualisations retenus.
5. Signaler toute hypothèse ou donnée manquante.

Puis :
1. Implémenter la simulation complète.
2. Fournir les fichiers modifiés dans leur intégralité ou sous forme de patch clair.
3. Donner les commandes d'installation et de lancement sous Windows et/ou Linux.
4. Vérifier les imports, chemins, cas où les données sont absentes, et cohérence des métriques.
5. Fournir un court guide pour présenter et expliquer la simulation à l'oral.

## 10. Référence d'inspiration : dossier `45`

Le dossier `45` contient des captures d'écran organisées par algorithme : `KNN`, `arbre de decision`, `Bayes` et `svm`. Il doit être utilisé comme **référence pédagogique et visuelle seulement** :
- s'inspirer des panneaux explicatifs, des visualisations de frontières/décisions, des contrôles interactifs et des comparaisons ;
- garder une interface originale adaptée aux variables de candidats et au pipeline réel du projet ;
- ne pas copier les captures, le code, le texte ou le design exact ;
- ne pas reprendre les points synthétiques de l'exemple comme s'ils étaient les candidats du dataset.

## 11. Décision de conception la plus importante

La simulation ne doit pas seulement montrer des frontières artificielles sur deux variables inventées. Elle doit relier chaque visualisation à une **prédiction réelle du pipeline du projet**, tout en indiquant honnêtement lorsqu'une projection en 2D simplifie un espace de données plus complexe.

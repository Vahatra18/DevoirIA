# Projet IA : Prédiction de la qualité du vin

Ce projet utilise **Pandas**, **TensorFlow**, **scikit-learn**, et **matplotlib** pour entraîner un modèle de classification supervisée afin de prédire la qualité du vin rouge.

## 📂 Dataset utilisé
Fichier CSV : `winequality-red.csv` provenant de l'archive UCI Machine Learning Repository.

## 🔧 Étapes principales
- Chargement et prétraitement des données
- Normalisation avec `StandardScaler`
- Entraînement d’un modèle **Keras (Dense Neural Network)**
- Évaluation avec une **matrice de confusion**
- Affichage des **courbes d’apprentissage**
- Prédiction sur une nouvelle donnée

## 📊 Résultats
- Précision atteinte sur l'ensemble de test : variable selon les paramètres.
- Matrice de confusion affichée pour analyse des prédictions.

## 📁 Fichier de code principal
```python
fichier_code.py  # contient tout le pipeline IA

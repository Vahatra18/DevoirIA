# Projet IA : Prédiction de la qualité du vin

Ce projet utilise **Pandas**, **TensorFlow**, **scikit-learn**, et **matplotlib** pour entraîner un modèle de classification supervisée afin de prédire la qualité du vin rouge.

## 📂 Dataset utilisé

Fichier CSV : `winequality-red.csv` provenant de l'archive UCI Machine Learning Repository.  
Chaque ligne représente un échantillon de vin rouge avec 11 caractéristiques (acidité, sucre, pH, alcool, etc.) et une note de qualité (de 3 à 8).

## 🔧 Étapes principales

- Chargement et prétraitement des données
- Normalisation avec `StandardScaler`
- Entraînement d’un modèle **Keras (Dense Neural Network)**
- Évaluation avec une **matrice de confusion**
- Affichage des **courbes d’apprentissage (perte & précision)**
- Prédiction sur une nouvelle donnée

## 📊 Résultats

- **Précision sur l'ensemble de test** : généralement entre **60% et 70%**, selon l'architecture du réseau et les hyperparamètres.
- **Matrice de confusion** : permet d’identifier les classes les plus confondues entre elles.
- L'entraînement est suivi de courbes pour analyser visuellement la **convergence** et détecter un éventuel surapprentissage.

## 📁 Fichier de code principal

```python
fichier_code.py  # contient tout le pipeline IA
```

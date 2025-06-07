import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Étape 1 : Définir le chemin du fichier
filename = r"E:\ProjetIA\winequality-red.csv"

# Vérifier si le fichier existe
if not os.path.exists(filename):
    print(f"Erreur : Le fichier '{filename}' n'est pas trouvé.")
    print("Vérifiez que 'winequality-red.csv' est dans 'E:\\ProjetIA'.")
    exit(1)

# Charger le dataset
try:
    data = pd.read_csv(filename, sep=';')  # Séparateur : ';'
except Exception as e:
    print(f"Erreur lors du chargement du fichier : {e}")
    print("Vérifiez que le fichier utilise des points-virgules comme séparateur.")
    exit(1)

# Étape 2 : Explorer les données
print("Aperçu des données :")
print(data.head())
print("\nDistribution des classes :")
print(data['quality'].value_counts())

# Étape 3 : Prétraitement
# Séparer features et cible
X = data.drop('quality', axis=1)
y = data['quality'] - 3  # Ajuster les labels (3-8 -> 0-5)

# Normaliser les features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Créer le modèle TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 classes
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Afficher un résumé du modèle
model.summary()

# Étape 5 : Entraîner le modèle
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1)

# Étape 6 : Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrécision sur l'ensemble de test : {accuracy:.4f}")

# Prédictions
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(3, 9), yticklabels=range(3, 9))
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()

# Courbes d'apprentissage
plt.figure(figsize=(12, 4))

# Perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perte (entraînement)')
plt.plot(history.history['val_loss'], label='Perte (validation)')
plt.title('Perte au fil des epochs')
plt.xlabel('Epoch')
plt.ylabel('Perte')
plt.legend()

# Précision
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Précision (entraînement)')
plt.plot(history.history['val_accuracy'], label='Précision (validation)')
plt.title('Précision au fil des epochs')
plt.xlabel('Epoch')
plt.ylabel('Précision')
plt.legend()

plt.tight_layout()
plt.show()

# Étape 7 : Tester sur une nouvelle donnée
print("\nTest sur une nouvelle donnée :")
new_wine = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]  # Exemple
new_wine_scaled = scaler.transform(new_wine)  # Normaliser
prediction = model.predict(new_wine_scaled).argmax(axis=1) + 3  # Reconvertir en qualité
print(f"Qualité prédite pour le nouveau vin : {prediction[0]}")
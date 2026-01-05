import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Chargement du fichier
# Pandas va automatiquement reconnaître la virgule comme séparateur
df = pd.read_csv('openaq_location_6133773_measurments(2).csv')

# Sélection des colonnes utiles
# On prend datetimeLocal (plus parlant pour l'activité humaine) et value
df = df[['datetimeLocal', 'value']]

# Conversion du format de date
df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])

# On renomme pour plus de clarté
df.columns = ['date', 'pm25']

# Tri chronologique (essentiel pour les séries temporelles)
df = df.sort_values('date')

# Suppression des doublons potentiels
df = df.drop_duplicates(subset=['date'])

print(f"Nombre de mesures : {len(df)}")
#print(df.head())

# Extraction de caractéristiques numériques
df['heure'] = df['date'].dt.hour
df['jour_semaine'] = df['date'].dt.dayofweek # 0=Lundi, 6=Dimanche

# Création du "LAG" (La valeur d'il y a 1 heure)
# C'est l'étape la plus importante : pour prédire le futur,
# on donne au modèle la valeur passée immédiate.
df['pm25_lag1'] = df['pm25'].shift(1)

# On supprime la première ligne qui a maintenant un "NaN" (vide) à cause du décalage
df = df.dropna()

print("\nTableau prêt pour l'IA :")
print(df.head())


#Modèle : entrainement & test

# Définition des variables explicatives (X) et de la cible (y)
# On utilise l'heure et la valeur précédente pour prédire la valeur actuelle
X = df[['heure', 'jour_semaine', 'pm25_lag1']]
y = df['pm25']

# Séparation Entraînement / Test
# IMPORTANT : En série temporelle, on ne mélange pas les données (shuffle=False)
# On apprend sur le passé (80%) pour prédire le futur (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Entraînement sur {len(X_train)} lignes, Test sur {len(X_test)} lignes.")

# Création et entraînement du modèle
# On utilise Random Forest, très efficace pour capter les relations non-linéaires
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur les données de test
predictions = model.predict(X_test)

# Calcul de l'erreur moyenne
mae = mean_absolute_error(y_test, predictions)
print(f"\nErreur Moyenne Absolue (MAE) : {mae:.2f} µg/m³")

# Coefficient de détermination
score = r2_score(y_test, predictions)
print(f"Efficacité du modèle (R²) : {score*100:.2f}%")

# Analyse de l'importance des variables
importances = model.feature_importances_
for i, feat in enumerate(X.columns):
    print(f"Influence de '{feat}' sur la prédiction : {importances[i]*100:.1f}%")

# Visualisation des résultats
plt.figure(figsize=(14, 7))
plt.plot(y_test.values[:100], label="Réel (Vérité terrain)", color='#2ecc71', linewidth=2)
plt.plot(predictions[:100], label="Prédiction IA", color='#e74c3c', linestyle='--', linewidth=2)
plt.title("Test du modèle : Prédiction vs Réalité (100 premières heures du test set)")
plt.xlabel("Temps (Heures)")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
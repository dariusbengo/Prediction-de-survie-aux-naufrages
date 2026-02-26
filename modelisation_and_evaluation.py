# %%
import pandas as pd
naufrages_clean = pd.read_csv("https://raw.githubusercontent.com/dariusbengo/Prediction-de-survie-aux-naufrages/main/data/processed/naufrages_clean.csv")
naufrages_clean = naufrages_clean.drop(columns=['Naufrage','Category'])

# %% [markdown]
# Encodage des variables avec OneHotEncoder

# %%
from sklearn.preprocessing import OneHotEncoder
ohe_pas = OneHotEncoder(drop='if_binary', sparse_output=False)
passenger_encoded = pd.DataFrame(
    ohe_pas.fit_transform(naufrages_clean[['Passenger/Crew']]),
    columns=ohe_pas.get_feature_names_out()
)

naufrages_clean = pd.concat([naufrages_clean.drop('Passenger/Crew', axis=1), passenger_encoded], axis=1)
naufrages_clean

# %%
#Corrélations
naufrages_corr=naufrages_clean.corr(numeric_only=True)
naufrages_corr["Survived"].sort_values(ascending=False)

# %% [markdown]
# Les valeurs les plus corrélés à la probabilité de survivre sont `Hauteur_mer`, `Beaufort`, `Age`, et `Jour`.

# %% [markdown]
# Normalisation des données

# %%
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
naufrages_standardized=scaler.fit_transform(naufrages_clean)


# %% [markdown]
# ##### Création des échantillons

# %%
total_rows = 3830
training=total_rows*0.7
test_valid=total_rows*0.3
print("70%=",training, "\n30%=",test_valid)

# %% [markdown]
# Variable a predire : y = `Survived`
# 
# Variables predicitives :  X = les autres features sauf `Survived`
# 
# Training contient 70% des données 
# 
# Validation contient 15% des données 
# 
# Test contient 15% des données 

# %%
from sklearn.model_selection import train_test_split

y=naufrages_clean["Survived"]
X = naufrages_clean.drop(["Survived"], axis=1)

# %% [markdown]
# Split en 3 sets de données: training, validation, test
# Etape 1 : training = 70% et temporairement (30% --> validation + test)

# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# Etape 2 : validation (15%) et test (15%)

# %%
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# %% [markdown]
# On a les 3 sets de données : X_train + y_train | X_val + y_val | X_test + y_test

# %% [markdown]
# On normalise les 3 sets de X

# %%
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled =scaler.transform(X_val)
X_test_scaled =scaler.transform(X_test)

# %% [markdown]
# On va comparer plusieurs modèles pouvoir lequel a la meilleur performance. On va utiliser la Regression logisitque, le Support Vector Machine (SVM), le RandomForest et le GradientBoost.
# 
# <u>Mesure de la performance avec :</u> 
#  
#  F1-Score --> donne le part de prédictions de survie correcte
#  
#  ROC-AUC --> donne le part de prédiction de surive supérieure à la part de prédiction de non-survie
#  
#  Accuracy --> donne la part de bonnes prédictions

# %% [markdown]
# ##### Régression logistique
# 
# 

# %%

from sklearn.linear_model import LogisticRegression
import numpy as np
#apprentissage du modele
model=LogisticRegression()
model.fit(X_train_scaled,y_train)
#prediction
y_val_pred = model.predict(X_val_scaled)
y_proba = model.predict_proba(X_val_scaled)[:,1]

from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

val_f1 = f1_score(y_val, y_val_pred)
val_roc = roc_auc_score(y_val, y_proba)
val_acc=balanced_accuracy_score(y_val, y_val_pred)

print("Performance Regression linéaire :\nF1-Score :",val_f1,"\nROC-AUC :",val_roc,"\nAccuracy :",val_acc)


# %% [markdown]
# ##### SVM

# %%
from sklearn.svm import SVC
#apprentissage du modele
model2=SVC(probability=True)
model2.fit(X_train_scaled,y_train)
#prediction
y_val_pred2 = model2.predict(X_val_scaled)
y_proba2= model2.predict_proba(X_val_scaled)[:,1]

from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

val_f1_2 = f1_score(y_val, y_val_pred2)
val_roc_2 = roc_auc_score(y_val, y_proba2)
val_acc_2=balanced_accuracy_score(y_val, y_val_pred2)

print("Performance SVM:\nF1-Score :",val_f1_2,"\nROC-AUC :",val_roc_2,"\nAccuracy :",val_acc_2)


# %% [markdown]
# ##### RandomForest

# %%
from sklearn.ensemble import RandomForestClassifier
#apprentissage du modele
model3=RandomForestClassifier(n_estimators=300, random_state=42) #plus le nombre d'arbres est élevé, plus le temps de calcul sera long
model3.fit(X_train_scaled,y_train)
#prediction
y_val_pred3 = model3.predict(X_val_scaled)
y_proba3= model3.predict_proba(X_val_scaled)[:,1]

from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

val_f1_3 = f1_score(y_val, y_val_pred3)
val_roc_3 = roc_auc_score(y_val, y_proba3)
val_acc_3=balanced_accuracy_score(y_val, y_val_pred3)

print("Performance RandomForest:\nF1-Score :",val_f1_3,"\nROC-AUC :",val_roc_3,"\nAccuracy :",val_acc_3)


# %% [markdown]
# ##### GradientBoost
# 
# 

# %%
from sklearn.ensemble import GradientBoostingClassifier
#apprentissage du modele
model4=GradientBoostingClassifier(n_estimators=450, learning_rate=0.25, random_state=42) 
model4.fit(X_train_scaled,y_train)
#prediction
y_val_pred4 = model4.predict(X_val_scaled)
y_proba4= model4.predict_proba(X_val_scaled)[:,1]
print("Part d'explication de chaque feature dans la prédiction de survie :\n")
n_features = X_train_scaled.shape[1]
for i in range (n_features):
    print(X_train.columns.tolist()[i],":",model4.feature_importances_[i])
#performance

from sklearn.metrics import mean_squared_error, r2_score

val_f1_4 = f1_score(y_val, y_val_pred4)
val_roc_4 = roc_auc_score(y_val, y_proba4)
val_acc_4=balanced_accuracy_score(y_val, y_val_pred4)

print("\nPerformance GradientBoosting:\nF1-Score :",val_f1_4,"\nROC-AUC :",val_roc_4,"\nAccuracy :",val_acc_4)


# %% [markdown]
# Benchmark des performances des modèles

# %% [markdown]
# Dans les 3 métriques de performance, il y a le même classement : GradientBoost > RandomForest > SVM > LogisticRegression
# 
# Le Gradient Boost est donc le meilleur modèle pour prédire la survie d'une personne lors d'un naufrage. Il prédit 72,6% de bonnes réponses avec comme paramètres : 450 estimateurs, un taux d'apprentissage de 0.2 et un random_state de 42.

# %%


# Feature importance TOP 5
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model4.feature_importances_
}).sort_values('Importance', ascending=False).head()

print("\n Features les plus importantes:")
print(importances)

# Sauvegarde du modele
import joblib
joblib.dump(model4, 'gradientboosting_final.pkl')
joblib.dump(scaler, 'scaler_final.pkl')



# %%




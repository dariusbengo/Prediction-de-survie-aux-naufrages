# %% [markdown]
# # Prédiction de survie aux naufrages

# %% [markdown]
# ### Partie 1 : Collecte des données 

# %% [markdown]
# On va importer les données de la liste des passagers du Titanic via Kaggle

# %%
from pathlib import Path
import urllib.request
import os 
import pandas as pd

def github_data(file_path):
    csv_path=Path(f"datasets/{file_path}")
    if not csv_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url=f"https://raw.githubusercontent.com/dariusbengo/prediction-de-survie-aux-naufrages/main/data/raw/{file_path}"
        urllib.request.urlretrieve(url,csv_path)
    return pd.read_csv(csv_path)


titanic=github_data("Titanic-Dataset.csv")
lusitania=github_data("RMS%20Lusitania%20-%20Sheet1.csv")
estonia= github_data("estonia-passenger-list.csv")



# %% [markdown]
# On explore les bases de données

# %%
titanic

# %%
lusitania

# %%
estonia

# %%
titanic.describe()


# %%
lusitania.describe()


# %%
estonia.describe()

# %% [markdown]
# ### Nettoyage des bases
# 
# En analysant ces bases de données on remarque que l'on peut dès à présent supprimer certaines variables qui ne seront pas exploitables.
# 
# 

# %% [markdown]
# Pour la base de données du Titanic on va retirer les variables relatives à l'identité des passagers (PassengerID, Name), au billet (Ticket, Fare, Cabin), et au lieu d'embarquement (Embarked). Elles ne donnent pas d'information exploitables par l'ensemble des données.

# %%
titanic1=titanic[["Survived","Pclass","Sex", "Age", "SibSp","Parch"]]
titanic1

# %% [markdown]
# On pourra donc regrouper les données grâce aux variables relavtives à l'age, au sexe, au statut de la personne (passager ou membre de l'équipe du bateau), s'il est accompagné.
# 
# On va standardiser la manière de présenter le sexe. 

# %%
titanic1.loc[titanic1["Sex"] == "male", "Sex"] = "M"
titanic1.loc[titanic1["Sex"] == "female", "Sex"] = "F"

# %% [markdown]
# Cependant, on remarque qu'il y a des passagers dont l'age est inconnu.

# %%

age_inconnu=titanic1["Age"].isna().sum()/titanic1["Age"].count()
print("Il y a", f"{age_inconnu:2%}", "de personnes ayant un age inconnu" )

# %% [markdown]
# On sait désormais qu'il y a 25% de personnes dont l'âge est inconnu.
# 
# C'est un chiffre conséquent donc il ne serait pas idéal de retirer des données les personnes dont l'âge est inconnu, afin de ne pas biaiser les performances des modèles que nous mettrons en place. 
# 
# On va remplacer les valeurs manquantes par la mediane conditionnée au fait d'avoir survécu, au sexe, à la classe du passager et si le passager est marié ou non.
# 
# 
# 

# %%
age_groupe_median = (titanic1.groupby(["Survived","Sex", "Pclass","SibSp"])["Age"].transform("median"))

titanic1["Age"] = titanic1["Age"].fillna(age_groupe_median)
titanic1

# %% [markdown]
# Je crée une colonne "Accompagné" qui vaudra 1 si le passager a ses variabels SibSp et Parch supérieures à 0 car cela signifie qu'il est accompagné.

# %%
titanic1["Accompagné"] = ((titanic1["SibSp"] > 0) | (titanic1["Parch"] > 0)).astype(int)

# %% [markdown]
# Sachant que toutes les personnes de cette liste sont des passagers, je rajoute une colonne "Passenger" pù je le précise explicitement. Cela pourra améliorer les performances lors de l'apprentissage.

# %%
titanic1["Passenger/Crew"]="Passenger"

# %% [markdown]
# J'ajoute une dernière colonne nommée "Naufrage" où je précise que tous ces passagers ont vécu le naufrage du Titanic. Cela pourra améliorer l'apprentissage des modèles.

# %%
titanic1["Naufrage"]="Titanic"

# %% [markdown]
# Dans la base de données du neaufrage du RMS Lusitania, on va conserver uniquement les variables qui donnent les mêmes informations que la base de données nettoyée du Titanic.
# 
# On retient donc le destin de la personne, son sexe, son age, sa classe, si elle était un passager ou un membre de l'équipe et si elle avait des accompagnateurs.
# 
# On va standardiser la manière de présenter le sexe.

# %%
lusitania1=lusitania[["Fate","Sex","Age","Department/Class","Passenger/Crew"]]
lusitania1.loc[lusitania1["Sex"] == "Male", "Sex"] = "M"
lusitania1.loc[lusitania1["Sex"] == "Female", "Sex"] = "F"

# %% [markdown]
# Il y a des données sur l'âge qui ne sont pas au format numérique. Il y a par exemple des "49 ?" ou des "13-months". On va mettre ces âges au bon format. 

# %%
#Correction des âges au mauvais format

import re

def clean_age(x):
    if pd.isna(x):
        return None
    x = str(x).strip()

    # Cas du type "49 ?" -> garder 49
    m = re.match(r"(\d+)", x)
    if m:
        val = int(m.group(1))
    else:
        return None

    # Cas des mois : "13-months", "05-months" -> convertir en années
    if "month" in x.lower():
        return val / 12.0
    else:
        return float(val)

lusitania1["Age"] = lusitania1["Age"].apply(clean_age)

# %% [markdown]
# On va modifier la variable "Fate" pour la rendre binaire.
# 
# 0 : Lost
# 1 : Saved, Saved (died from trauma), Not on board

# %%
def survived(survived):
    if survived in ['Lost']:  # 0 = Mort
        return 0
    else:  # 1 = Survivant (Saved, 1, etc.)
        return 1

lusitania1['Survived'] = lusitania1['Fate'].apply(survived)

# %% [markdown]
# On rajoutera également une colonne "accompagné" pour transposer les données textuelles de "Traveling Companions and other notes" en données binaires valant 1 si la personne est accompagnée ou 0 sinon.

# %%

lusitania1["Accompagné"]=lusitania["Traveling Companions and other notes"].notna().astype(int)

#dictionnaire pour définir les nouvelles classes
Classes = {
    "Saloon": "1", #1ere classe
    "Deck": "1",
    "Second": "2", #2nde classe
    "Third": "3", #3eme classe
    "Stowaway": "3",
    "Third (Distressed British Seaman)": "3",
    "Band": "4", #employés de navire
    "Engineering": "4",
    "Victualling": "4"}

lusitania1["Department/Class"] = (lusitania1["Department/Class"].replace(Classes))

lusitania1

# %% [markdown]
# On va effectuer la même méthode pour les données manquantes sur l'âge que ce que l'on a effcectué sur les données de la base du Titnaic. 

# %%


age_groupe_median_2 = (lusitania1.groupby(["Survived","Sex", "Department/Class","Passenger/Crew","Accompagné"])["Age"].transform("median"))

lusitania1["Age_theorique"] = lusitania1["Age"].fillna(age_groupe_median_2)
lusitania1

# %% [markdown]
# J'ajoute une dernière colonne nommée "Naufrage" où je précise que tous ces passagers ont vécu le naufrage du RMS Lusitania. Cela pourra améliorer l'apprentissage des modèles.

# %%
lusitania1["Naufrage"]="Lusitania"

# %% [markdown]
# Pour la base de données du naufrage du MS Estonia, on va 

# %%
estonia1 = estonia.dropna(subset=['Age', 'Category'])  # lignes complètes
estonia1

# %% [markdown]
# Ici, Category ne prend que deux valeurs qui sont P et C, soit Passenger et Crew. On va remplacer P par Passenger et C par crew.

# %%
estonia1.loc[estonia1["Category"] == "C", "Category"] = "Crew"
estonia1.loc[estonia1["Category"] == "P", "Category"] = "Passenger"

# %% [markdown]
# Pour facilier la jointure avec les deux autres bases de données, on va essayer de supposer des liens de parenté entre les passagers du bateau. 
# 
# Si on trouve le même nom de famille et un âge similaire (dans une tranche d'âge de 10 ans), on suppose alors que ces personnes sont venues en groupe. la limite est que si les âfes sont 29 et 31 ans, ils ne seront pas considérés dans le même groupe.

# %%
import numpy as np

# On crée des tranches d'âge grossières par pas de 10 an
estonia1["Age_bin"] = (estonia["Age"] // 10).astype("Int64")  # 0: 0-9, 1: 10-19, etc.

# On construit un identifiant de groupe 
estonia1["lastname_ageblock"] = (estonia["Lastname"].astype(str) + "_" + estonia1["Age_bin"].astype(str))

# On compte combien il y a de personnes par groupe
counts = estonia1["lastname_ageblock"].value_counts()

# On crée l'indicateur "Accompagné" : 1 si au moins 2 personnes dans le même groupe
estonia1["Accompagné"] = estonia1["lastname_ageblock"].map(lambda x: 1 if counts[x] > 1 else 0).astype(int)

# %% [markdown]
# J'ajoute une dernière colonne nommée "Naufrage" où je précise que tous ces passagers ont vécu le naufrage du RMS Lusitania. Cela pourra améliorer l'apprentissage des modèles.

# %%
estonia1["Naufrage"]="Estonia"

# %%
#on nettoie en enlevant les colonnes qui ne seront plus traitées par la suite
estonia1["Passenger/Crew"]=estonia1["Category"]
estonia_clean=estonia1[["Sex","Age","Passenger/Crew","Survived","Accompagné","Naufrage"]]

# %% [markdown]
# Nous avons effectué tout le nettoyage nécessaire de ces bases ainsi que créé des features essentielles pour l'apprentissage des modèles.

# %%
titanic1["Category"]=titanic1["Pclass"]
titanic_clean=titanic1[["Survived","Category", "Sex","Age","Accompagné","Passenger/Crew","Naufrage"]]
titanic_clean


# %%

lusitania1["Category"]=lusitania1["Department/Class"]
lusitania_clean=lusitania1[["Survived","Sex","Age","Category","Passenger/Crew","Accompagné","Naufrage"]]
lusitania_clean["Age"]=lusitania1["Age_theorique"]
lusitania_clean


# %%
estonia_clean

# %%
#Téléchargement des bases de données "processed"
import os
os.makedirs("C:/Users/lenovo/OneDrive/Documents/M2/Machine Learning/data/processed", exist_ok=True)

titanic_clean.to_csv("C:/Users/lenovo/OneDrive/Documents/M2/Machine Learning/data/processed/titanic_clean.csv", index=False)
lusitania_clean.to_csv("C:/Users/lenovo/OneDrive/Documents/M2/Machine Learning/data/processed/lusitania_clean.csv", index=False)
estonia_clean.to_csv("C:/Users/lenovo/OneDrive/Documents/M2/Machine Learning/data/processed/estonia_clean.csv", index=False)


# %%




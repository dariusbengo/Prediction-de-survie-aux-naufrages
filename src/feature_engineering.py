# %%
#Chargement des tables clean
import pandas as pd
titanic_clean = pd.read_csv("https://raw.githubusercontent.com/dariusbengo/Prediction-de-survie-aux-naufrages/main/data/processed/titanic_clean.csv")
lusitania_clean = pd.read_csv("https://raw.githubusercontent.com/dariusbengo/Prediction-de-survie-aux-naufrages/main/data/processed/lusitania_clean.csv")
estonia_clean = pd.read_csv("https://raw.githubusercontent.com/dariusbengo/Prediction-de-survie-aux-naufrages/main/data/processed/estonia_clean.csv")



# %% [markdown]
# ### Ajout de données externes
# 
# ##### Conditions météorologiques
# 
# Nous allons ajouter les variables suivantes :
# 
#  `Jour/Nuit` valant 1 si le naufrage au lieu pendant la journée et 0 sinon.
# 
#  `Beaufort` allant de 0 à 12. L'échelle de Beaufort permet de mesurer la vitesse moyenne du vent sur une durée de dix minutes utilisée dans les milieux maritimes.
# 
#  `Hauteur de la mer` qui prend des valeurs numériques (en mètre).
# 
# Lors du naufrage du Titanic, d'après les rapports historiques, cela s'est déroulé la nuit, la mer était très calme, il y avait une bonne visibilité, peu de vent et une présence massive d’icebergs.

# %%
titanic_clean["Jour/Nuit"]=0
titanic_clean["Beaufort"]=1
titanic_clean["Hauteur_mer"]=0.1

lusitania_clean["Jour/Nuit"]=1
lusitania_clean["Beaufort"]=3
lusitania_clean["Hauteur_mer"]=0.6

estonia_clean["Jour/Nuit"]=0
estonia_clean["Beaufort"]=8
estonia_clean["Hauteur_mer"]=5.5




# %% [markdown]
# On va regrouper les 3 tables nettoyées ensemble 

# %%
naufrages = pd.concat([titanic_clean, lusitania_clean, estonia_clean], ignore_index=True)
naufrages


# %%
naufrages["Survived"].value_counts()

# %%
naufrages.info()

# %%
naufrages1=naufrages.dropna(subset="Age")

# %%
naufrages1.info()

# %%
# transformation des variables en booléen
naufrages1['Survived'] = naufrages1['Survived'].astype(int)  

naufrages1['Accompagné'] = (naufrages1['Accompagné'] > 0).astype(int)
naufrages1['Sex_male'] = (naufrages1['Sex'] == 'M').astype(int)  
naufrages1['Jour'] = (naufrages1['Jour/Nuit'] == 1).astype(int)  


#transformation en categories 
naufrages1['Category'] = naufrages1['Category'].astype('category')
naufrages1['Passenger/Crew'] = naufrages1['Passenger/Crew'].astype('category')


# %%
naufrages_clean=naufrages1[["Survived","Category","Sex_male","Age","Accompagné","Passenger/Crew","Naufrage","Jour","Beaufort","Hauteur_mer"]]
naufrages_clean

# %%
naufrages_clean.info()

# %%
naufrages_clean.to_csv("C:/Users/lenovo/OneDrive/Documents/M2/Machine Learning/data/processed/naufrages_clean.csv", index=False)



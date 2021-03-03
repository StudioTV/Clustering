import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Tuple
import numpy as np
import csv

def load_dataset(pathname:str) -> Tuple[np.ndarray, np.ndarray]:
    # check the file format through its extension
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    # open the file in read mode
    with open(pathname, 'r') as csvfile:
        # create the reader object in order to parse the data file
        reader = csv.reader(csvfile, delimiter=',')
        # extract the data and the associated label
        # (he last column of the file corresponds to the label)
        data = []
        labels = []
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
        # converts Python lists into NumPy matrices
        # in the case of the list of labels, generate an int id per class
        data = np.array(data, dtype=np.float)
        lookupTable, labels = np.unique(labels, return_inverse=True)
    # return data with the associated label
    return data, labels


filepath = "./train.csv" # Le chemin vers votre fichier
data, labels = load_dataset(filepath) # Chargement des données dans les variables data et labels
plt.scatter(data[:,0],data[:,1],s=30) # Création du graphe

plt.show() # Affichage des données

kmeans = KMeans(n_clusters=3).fit(data) # Utilisation du k-means sur nos données

print(kmeans.cluster_centers_) # Affichage de nos centroides

kmeans_pred = kmeans.predict(data) # Prediction sur nos données

plt.scatter(data[:,0],data[:,1],c=kmeans_pred, s=30,cmap='rainbow') # Création du graphe avec les clusters
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') # Création des points du centre des clusters en noir

plt.show() # Affichage final
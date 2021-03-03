import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits() # Chargement du dataset
X_digits, y_digits = digits.data, digits.target # X nos données  Y la vrai valeur de la donnée

estimator = PCA(n_components=10) #10 composants 0-9 
X_pca = estimator.fit_transform(X_digits) # Fit sur nos données

colors = ['black', 'blue', 'purple', 'yellow', 'white','red', 'lime', 'cyan', 'orange', 'gray'] # Tableau de couleur

for i in range(len(colors)):
	px = X_pca[:, 0][y_digits == i] #coord x
	py = X_pca[:, 1][y_digits == i] #coord y
	plt.scatter(px, py, c=colors[i])  # on assigne une couleur au cluster
	plt.legend(digits.target_names)
	plt.xlabel('Première composante principale')
	plt.ylabel('Deuxième composante principale')
		
plt.show() # Affichage final

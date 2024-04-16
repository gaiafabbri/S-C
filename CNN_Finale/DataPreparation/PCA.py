import os
import numpy as np
from sklearn.decomposition import PCA

'''------------------Principal Component Analysis-----------------------'''
'''Introducing Principal Component Analysis to deal with a large dimensional dataset
1)The PCA nees a a one-dimensional input, so the images are taken as one dimensional arrays; the PCA is performed on normalised data
2)The number of principal components is computed to achieve the 95% of the total variance
3) Applying the PCA allows to reduce in dimensionality the dataset
'''

def find_optimal_num_components(normalized_data, target_variance_explained, width, height):
    # Calcolo della PCA, dove n_components è la dimensione dell'immagine
    pca = PCA(n_components=width*height)
    pca.fit(normalized_data)
    pca_result_data = pca.transform(normalized_data)
    # Calcolo della varianza cumulativa
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Trova il numero di componenti che spiegano una percentuale sufficiente della varianza
    num_components = np.argmax(cumulative_variance_ratio >= target_variance_explained) + 1

    return num_components

def apply_pca(normalized_data, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(normalized_data)
    pca_result = pca.transform(normalized_data)
    return pca_result

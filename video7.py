import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# Preparando datos de prueba
X, y, c = make_blobs(n_samples=500, cluster_std=0.8, centers=4, n_features=2, return_centers=True)

# Crear un DataFrame con los datos generados (puntos de los clusters)
df_blobs = pd.DataFrame(
    {
        'x1': X[:, 0],
        'x2': X[:, 1],
        'y': y
    }
)

# Crear un DataFrame con los centros de los clusters
df_centers = pd.DataFrame(
    {
        'x1': c[:, 0],
        'x2': c[:, 1]
    }
)

# Ejecutar KMeans con 3 clusters
kmeans = KMeans(n_clusters=3)
df_cluster = kmeans.fit_predict(X)

# Asignar los clusters al DataFrame de blobs
df_blobs['cluster'] = df_cluster

# Obtener los centros calculados por KMeans
k_means_centers = kmeans.cluster_centers_

# Crear un DataFrame con los centros de KMeans
df_K_means_center = pd.DataFrame(
    {
        'x1': k_means_centers[:, 0],
        'x2': k_means_centers[:, 1]
    }
)

# Visualización de los datos
fig = plt.figure(figsize=(9, 9))

# Graficar los puntos de los clusters
sns.scatterplot(data=df_blobs, x='x1', y='x2', hue='cluster', palette='coolwarm')

# Graficar los centros originales
sns.scatterplot(data=df_centers, x='x1', y='x2', marker='X', s=150, color='red', label='Centros Originales')

# Graficar los centros de KMeans
sns.scatterplot(data=df_K_means_center, x='x1', y='x2', marker='o', s=150, color='yellow', label='Centros KMeans')

# Mostrar la gráfica
plt.show()

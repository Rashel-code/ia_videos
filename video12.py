import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Generar los datos de prueba
X, y, c = make_blobs(n_samples=500, cluster_std=0.8, centers=4, n_features=2, return_centers=True)

# Crear un DataFrame con los datos generados (puntos)
df_blobs = pd.DataFrame(
    {
        'x1': X[:, 0],  # Primera característica de los puntos
        'x2': X[:, 1],  # Segunda característica de los puntos
        'y': y  # Etiquetas de los clusters originales
    }
)

# Crear un DataFrame con los centros de los clusters generados
df_centers = pd.DataFrame(
    {
        'x1': c[:, 0],  # Coordenada x1 de los centros de los clusters
        'x2': c[:, 1]   # Coordenada x2 de los centros de los clusters
    }
)

# Graficar los puntos generados por make_blobs
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
sns.scatterplot(data=df_blobs, x='x1', y='x2')  # Graficar los puntos x1 y x2
plt.show()

# Dendrograma y clustering jerárquico
plt.figure(figsize=(10, 10))  # Tamaño del gráfico
dendrogram_plot = dendrogram(linkage(X, method='ward'))  # Generar el dendrograma usando el método Ward
plt.title('Dendrograma usando ward linkage')  # Título del gráfico
plt.xlabel('Cluster', fontsize=12)  # Etiqueta del eje X
plt.ylabel('Distancia Euclidiana')  # Etiqueta del eje Y
plt.show()

# Aplicar Agglomerative Clustering
# Reemplazamos affinity con metric='euclidean'
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')

# Predecir los clusters para cada punto utilizando el modelo ajustado
y_hc = hc.fit_predict(X)

# Mostrar los clusters predichos
print(y_hc)

# Asignar los clusters predichos al DataFrame df_blobs
df_blobs['cluster'] = y_hc

# Graficar los puntos coloreados por los clusters predichos
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
sns.scatterplot(data=df_blobs, x='x1', y='x2', hue='cluster', palette='coolwarm')  # Graficar los puntos coloreados por clusters
plt.show()

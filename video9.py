import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# Generar datos de prueba
X, y, c = make_blobs(n_samples=500, cluster_std=0.8, centers=4, n_features=2, return_centers=True)

# Crear un DataFrame con los datos generados (puntos de los clusters)
df_blobs = pd.DataFrame(
    {
        'x1': X[:, 0],
        'x2': X[:, 1],
        'y': y
    }
)

# Crear un DataFrame con los centros de los clusters originales
df_centers = pd.DataFrame(
    {
        'x1': c[:, 0],
        'x2': c[:, 1]
    }
)

# Función para aplicar KMeans y visualizar los resultados
def vis_cluster(k):
    # Ejecutar KMeans con k clusters
    kmeans = KMeans(n_clusters=k)
    df_cluster = kmeans.fit_predict(X)

    # Asignar los clusters predichos al DataFrame
    df_blobs['cluster'] = df_cluster

    # Obtener los centros calculados por KMeans
    k_means_centers = kmeans.cluster_centers_

    # Crear un DataFrame con los centros calculados por KMeans para visualizarlos
    df_K_means_center = pd.DataFrame(
        {
            'x1': k_means_centers[:, 0],
            'x2': k_means_centers[:, 1]
        }
    )

    # Iniciar la visualización de los resultados con un gráfico de tamaño 9x9
    fig = plt.figure(figsize=(9, 9))

    # Graficar los puntos generados con los clusters asignados por KMeans
    sns.scatterplot(data=df_blobs, x='x1', y='x2', hue='cluster', palette='coolwarm')

    # Graficar los centros originales de los clusters en color rojo
    sns.scatterplot(data=df_centers, x='x1', y='x2', marker='X', s=150, color='red', label='Centros Originales')

    # Graficar los centros predichos por KMeans en color amarillo
    sns.scatterplot(data=df_K_means_center, x='x1', y='x2', marker='o', s=150, color='yellow', label='Centros KMeans')

    # Mostrar el gráfico
    plt.show()

# Bucle para visualizar los clusters con diferentes valores de k (de 3 a 6 clusters)
for k in range(3, 7):
    print(f'Visualizando con {k} clusters:')
    vis_cluster(k)

# 1. Visualización del método del "Elbow" (Codo)
sum_of_squared_distances = []
K = range(2, 15)  # Probar valores de k entre 2 y 14

# Aplicar KMeans para diferentes valores de k
for k in K:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sum_of_squared_distances.append(km.inertia_)  # Guardar la inercia (suma de distancias cuadradas)

# Graficar el resultado del método del "Elbow"
plt.figure(figsize=(8, 8))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia (Suma de Distancias Cuadradas)')
plt.title('Método del Elbow')
plt.show()

# 2. Visualización del Coeficiente de Silueta
silhouette_scores = []

# Probar valores de k entre 2 y 14
for k in K:
    km = KMeans(n_clusters=k)
    km.fit(X)
    y_pred = km.predict(X)  # Predecir los clusters
    silhouette_scores.append(silhouette_score(X, y_pred))  # Guardar el coeficiente de silueta

# Graficar el resultado del Coeficiente de Silueta
plt.figure(figsize=(8, 8))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silueta')
plt.title('Coeficiente de Silueta por Número de Clusters')
plt.show()

# 3. Visualización del Coeficiente de Silueta usando SilhouetteVisualizer (Yellowbrick)
# Este paso ayuda a visualizar cómo se distribuyen los clusters internamente con Silhouette Score.
plt.figure(figsize=(15, 8))  # Tamaño de la figura
km = KMeans(n_clusters=4)  # Aplicar KMeans con 4 clusters

# SilhouetteVisualizer crea un gráfico de barras que muestra el coeficiente de silueta para cada punto
visualizer = SilhouetteVisualizer(km, colors='yellowbrick')

# Ajustar el visualizador con los datos X (puntos generados)
visualizer.fit(X)

# Mostrar el gráfico de coeficiente de silueta
visualizer.show()

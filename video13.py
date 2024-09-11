# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.cm as cm  # Para manejar mapas de colores
import matplotlib.pyplot as plt  # Para crear gráficos
import numpy as np  # Para manejar arrays numéricos

from sklearn.cluster import KMeans  # Algoritmo KMeans para clustering
from sklearn.datasets import make_blobs  # Para generar datos simulados
from sklearn.metrics import silhouette_samples, silhouette_score  # Para calcular coeficientes de silueta

# Generar los datos de ejemplo con make_blobs
# Se generan 500 puntos distribuidos en 4 clusters. Un cluster es más distinto
# y los otros 3 están más cercanos.
X, y = make_blobs(
    n_samples=500,  # Número de muestras
    n_features=2,  # Número de características (dimensiones)
    centers=4,  # Número de centros (clusters) a generar
    cluster_std=1,  # Desviación estándar de los clusters
    center_box=(-10.0, 10.0),  # Los clusters estarán en este rango
    shuffle=True,  # Barajar las muestras
    random_state=1,  # Semilla aleatoria para reproducibilidad
)

# Lista con diferentes números de clusters a probar
range_n_clusters = [2, 3, 4, 5, 6]

# Bucle para probar cada número de clusters en range_n_clusters
for n_clusters in range_n_clusters:
    # Crear un subplot con 1 fila y 2 columnas
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)  # Tamaño de la figura

    # Primer gráfico es el gráfico de silueta
    # El coeficiente de silueta puede estar entre -1 y 1. En este caso está entre [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # El (n_clusters+1)*10 es para insertar espacio en blanco entre los gráficos
    # de silueta de los clusters individuales.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Inicializar el modelo KMeans con el valor de n_clusters y una semilla aleatoria de 10
    # para garantizar la reproducibilidad.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)  # Ajustar el modelo y predecir los clusters

    # Calcular el coeficiente de silueta promedio para todas las muestras.
    # Esto da una perspectiva de la densidad y separación de los clusters formados.
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Calcular los valores de silueta para cada muestra (punto)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # Variable para controlar la posición en el gráfico de silueta
    y_lower = 10
    for i in range(n_clusters):
        # Agrupar los valores de silueta de las muestras que pertenecen al cluster i
        # y ordenarlos
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        # Tamaño del cluster i
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i  # Definir el límite superior de Y para el cluster

        # Asignar un color para este cluster
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),  # Posición en Y
            0,  # Inicia en 0 en el eje X
            ith_cluster_silhouette_values,  # Los valores de silueta
            facecolor=color,  # Color de fondo
            edgecolor=color,  # Color del borde
            alpha=0.7,  # Transparencia
        )

        # Etiquetar el gráfico de silueta con el número del cluster en su centro
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Calcular el nuevo valor de y_lower para el próximo cluster
        y_lower = y_upper + 10  # Añadir espacio entre los clusters

    # Configurar título y etiquetas del gráfico de silueta
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # Dibujar una línea vertical que indique el valor promedio de la silueta
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Limpiar las etiquetas del eje Y
    ax1.set_yticks([])  # Eliminar las marcas del eje Y
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])  # Definir las marcas del eje X

    # Segundo gráfico que muestra los clusters formados
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)  # Colorear los puntos según el cluster
    ax2.scatter(
        X[:, 0], X[:, 1],  # Coordenadas X e Y de los puntos
        marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"  # Estilo de los puntos
    )

    # Etiquetar los clusters
    centers = clusterer.cluster_centers_  # Obtener los centros de los clusters
    # Dibujar círculos blancos en los centros de los clusters
    ax2.scatter(
        centers[:, 0], centers[:, 1],  # Coordenadas de los centros
        marker="o",  # Marcador circular
        c="white",  # Color blanco para los centros
        alpha=1,  # Sin transparencia
        s=200,  # Tamaño de los círculos
        edgecolor="k",  # Borde negro
    )

    # Etiquetar cada centro de cluster con su número
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")  # Etiquetas numéricas en los centros

    # Configurar el título y etiquetas del gráfico de clusters
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")  # Etiqueta para el eje X
    ax2.set_ylabel("Feature space for the 2nd feature")  # Etiqueta para el eje Y

    # Título general de la figura
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

# Mostrar los gráficos
plt.show()

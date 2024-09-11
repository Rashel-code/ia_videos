from sklearn.datasets import make_blobs # crear el data set de datos
import pandas as pd #para manipular los dataser
import matplotlib.pyplot as plt #visualizacion de datos
import numpy as np
#n_samples= numeros de ejemplos 
#centers= crear cluters
#n_features= numero de features ( caracteristicas)
# cluster_std= desviacion estandar de clutes, que tan cerca estan
# random_state= para poder replicar los resultados
x, y = make_blobs(n_samples=100, centers=4, n_features=2, cluster_std=[1, 1.5, 2, 2], random_state=7)

# Crear un gráfico de dispersión
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', marker='o', s=50)

# Agregar etiquetas y título
plt.title("Gráfico de dispersión de los datos generados con make_blobs")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

# Mostrar el gráfico
plt.show()
print(np.array(y).reshape(-1, 10))

print('----------------------\n\n')

df_blobis = pd.DataFrame({
    'x1' : x[:,0],
    'x2' : x[:,1],
    'y':y
})

print(df_blobis)

print('----------------------\n\n')

#DIBUJAR NUESTOS DATOS
#unique solo los valores que tiene y
def plot_2d_clusters(x, y, ax):
    y_uniques = pd.Series(y).unique()  # Encuentra los valores únicos en y (los clusters)

    for cluster in y_uniques:
        # Filtra los puntos que pertenecen a este cluster
        cluster_points = x[y == cluster]

        # Crea un gráfico de dispersión para los puntos del cluster actual
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', marker=f'${cluster}$')

    # Configura el título y las etiquetas
    ax.set_title(f'{len(y_uniques)} Clusters')
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.legend()

# Ejemplo de uso:
fig, ax = plt.subplots()
plot_2d_clusters(x, y, ax)
plt.show()

print('----------------------\n\n')

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Inicialización del modelo KMeans con 4 clusters
kmeans = KMeans(n_clusters=4, random_state=7)

# Entrenar el modelo y predecir los clusters
y_pred = kmeans.fit_predict(x)

# Crear la figura y los subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 12))

# Graficar los clusters reales y predichos
plot_2d_clusters(x, y, axs[0])          # Clusters reales
plot_2d_clusters(x, y_pred, axs[1])      # Clusters predichos por KMeans

# Ajustar los títulos de los gráficos
axs[0].set_title(f'Actual {axs[0].get_title()}')
axs[1].set_title(f'KMeans {axs[0].get_title()}')

# Mostrar el gráfico
plt.show()


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Cargar el conjunto de datos Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Crear el modelo KMeans con los parámetros indicados
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0, n_init="auto")

# Ajustar el modelo a los datos
kmeans.fit(iris.data)

# Asignar las etiquetas de los clústeres al DataFrame
iris_df['cluster'] = kmeans.labels_

# Agrupar los datos por target y cluster, y contar el número de muestras en cada grupo
result = iris_df.groupby(['target', 'cluster']).agg({'sepal length (cm)': 'count'})
print(result)

# Realizar la reducción de dimensionalidad con PCA a 2 componentes principales
pca = PCA(2)
pca_res = pca.fit_transform(iris.data)

# Añadir los resultados de PCA al DataFrame
iris_df['X'] = pca_res[:, 0]
iris_df['Y'] = pca_res[:, 1]

# Dividir los datos por clúster para visualización
cluster_0 = iris_df[iris_df['cluster'] == 0]
cluster_1 = iris_df[iris_df['cluster'] == 1]
cluster_2 = iris_df[iris_df['cluster'] == 2]

# Graficar los resultados de K-means con las dos componentes principales de PCA
plt.scatter(cluster_0['X'], cluster_0['Y'], label='cluster 0')
plt.scatter(cluster_1['X'], cluster_1['Y'], label='cluster 1')
plt.scatter(cluster_2['X'], cluster_2['Y'], label='cluster 2')
plt.legend()
plt.title('Visualización del resultado de K-means con 2 componentes PCA')

plt.xlabel('X')
plt.ylabel('Y')
plt.show()

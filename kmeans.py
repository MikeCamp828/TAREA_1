#El programa funciona mediante kmeans para darnos la informacion y ademas generar un informe en HTML
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np

#Cargar el conjunto de datos Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

#Crear el modelo KMeans con los parámetros indicados
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0, n_init="auto")

#Ajustar el modelo a los datos
kmeans.fit(iris.data)

#Asignar las etiquetas de los clústeres al DataFrame
iris_df['cluster'] = kmeans.labels_

#Agrupar los datos por target y cluster, y contar el número de muestras en cada grupo
result = iris_df.groupby(['target', 'cluster']).agg({'sepal length (cm)': 'count'})
print(result)

#Realizar la reducción de dimensionalidad con PCA a 2 componentes principales
pca = PCA(2)
pca_res = pca.fit_transform(iris.data)

#Añadir los resultados de PCA al DataFrame
iris_df['X'] = pca_res[:, 0]
iris_df['Y'] = pca_res[:, 1]

#Graficar los resultados de K-means con las dos componentes principales de PCA
plt.scatter(iris_df['X'], iris_df['Y'], c=iris_df['cluster'], cmap='viridis', label=iris_df['cluster'])
plt.title('Visualización del resultado de K-means con 2 componentes PCA')
plt.xlabel('X - PCA1')
plt.ylabel('Y - PCA2')
plt.colorbar()
plt.savefig("kmeans_pca.png")
plt.show()

#Crear un reporte de Pandas Profiling
profile = ProfileReport(iris_df, title="Reporte del conjunto de datos Iris con KMeans", explorative=True)

#Guardar el reporte como archivo HTML
profile.to_file("reporte_kmeans.html")

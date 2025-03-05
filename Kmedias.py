import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#Jorge Ortiz Ceballos
#No. Control 22280692

# Cargar el archivo CSV
file_path = 'TablaCalificaciones.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Seleccionar las columnas de calificaciones para el análisis
calificaciones = data.iloc[:, 2:24]  # Columnas de calificaciones

# Normalizar los datos utilizando la desviación estándar
scaler = StandardScaler()
calificaciones_normalizadas = scaler.fit_transform(calificaciones)

# Inicializar los centroides con los primeros 6 y los siguientes 6 alumnos
initial_centroids = np.vstack([calificaciones_normalizadas[0:6].mean(axis=0),
                               calificaciones_normalizadas[6:12].mean(axis=0)])

# Aplicar el algoritmo k-means con los centroides iniciales
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, random_state=42)
kmeans.fit(calificaciones_normalizadas)

# Agregar los grupos al DataFrame original
data['Grupo'] = kmeans.labels_

# Mostrar los estudiantes en cada grupo
grupo_1 = data[data['Grupo'] == 0][['Nombre']]
grupo_2 = data[data['Grupo'] == 1][['Nombre']]

print("Grupo 1:")
print(grupo_1)
print("\nGrupo 2:")
print(grupo_2)

# Generar el gráfico de dispersión
plt.figure(figsize=(10, 6))

# Usar las dos primeras características para el gráfico
plt.scatter(calificaciones_normalizadas[data['Grupo'] == 0][:, 0],
            calificaciones_normalizadas[data['Grupo'] == 0][:, 1],
            c='blue', label='Grupo 1')

plt.scatter(calificaciones_normalizadas[data['Grupo'] == 1][:, 0],
            calificaciones_normalizadas[data['Grupo'] == 1][:, 1],
            c='red', label='Grupo 2')

# Graficar los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroides', marker='X')

plt.title('Clustering de Alumnos utilizando K-means')
plt.xlabel('Calificación Normalizada 1')
plt.ylabel('Calificación Normalizada 2')
plt.legend()
plt.show()

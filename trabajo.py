#Este codigo funciona para poder hacer reportes den HTML de forma automatica mediante un archivo CSV
#Importar las librerías necesarias
from ydata_profiling import ProfileReport 
import pandas as pd  

#Leer el archivo CSV y cargarlo en un DataFrame
df = pd.read_csv('iris.csv') 

#Generar un informe de perfilado de datos para el DataFrame
profile = ProfileReport(df, title="Profiling Iris Report")  

#Mostrar las primeras filas del DataFrame para inspección
df.head() 

#Mostrar el informe de perfilado de datos en un iframe dentro de un Jupyter Notebook
profile.to_notebook_iframe()  

#Guardar el informe de perfilado de datos en un archivo HTML
profile.to_file(".\my_report_iris.html")  

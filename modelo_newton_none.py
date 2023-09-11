# Importar librerias
# Para la base
import numpy as np
import pandas as pd

#Graficar
import seaborn as sns
import matplotlib.pyplot as plt

# Para el modelo
from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report


# Cargar el conjunto de datos
df = pd.read_csv('Iris.csv')

# No necesitamos la columna 'Id', así que la eliminamos
df = df.drop(['Id'], axis=1)

# Codificar las especies a valores numéricos
types = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['Species'] = df['Species'].replace(types)

# Mezclar el conjunto de datos para garantizar la aleatoriedad
df = shuffle(df)

# Dividir los datos en características (X) y variable objetivo (y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir y entrenar el modelo de regresión logística
model = LogisticRegression(solver='newton-cg', penalty='none', max_iter=100)
model.fit(X_train, y_train)
# Mostrar detalles del modelo
print(model.coef_)
print(model.intercept_)

# Predicciones
y_pred = model.predict(X_test)
y_predt = model.predict(X_train)
valores = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print(classification_report(y_test, y_pred, target_names=valores))
print(f'Exactitud del clasificador de regresión logística en test: {model.score(X_test, y_test)}')
print(f'Error del clasificador de regresión logística en test: {mean_squared_error(y_pred, y_test)}')
print(f'Error del clasificador de regresión logística en train: {mean_squared_error(y_predt, y_train)}')

# Gráficas
train_sizes, train_scores, test_scores = learning_curve(model, X, y)

# Media y STD para el entrenamiento
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Media y STD para la prueba
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

mat1 = confusion_matrix(y_test, y_pred)
sns.heatmap(mat1, annot=True, cmap='Blues', fmt='g', xticklabels=valores, yticklabels=valores)
plt.title('Matriz de Confusion')
plt.show()

plt.plot(train_sizes, train_mean, "o-", color="r", label='Goal Entrenamiento')
plt.plot(train_sizes, test_mean, "o-", color="g", label='Cross-validation')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)
plt.title('Curva Aprendizaje')
plt.legend(loc="best")
plt.grid()
plt.show()

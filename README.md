# M2_Uso_framework
Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)

Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)


INSTRUCCIONES PARA PODER CORRER EL CODIGO:

En este repositorio se encuentra un documento pdf, donde esta la documentación del proyecto el objetivo del algoritmo con respecto al dataset.

Y se encuentran tres archivos .py, en si los tres codigos tienen lo mismo, solamente hay una linea que cambia:

model = LogisticRegression(solver='---', penalty='---', max_iter=100)

La cual es donde se crea una instancia de un modelo de regresión logística utilizando la clase "LogisticRegression" de "sklearn.linear_model". 

El primero es con "liblinear", y "L1", el segundo con "newton-cg" y sin penalizacion, finalmente utilizo "sag" (Stochastic Average Gradient descent) y con "L2" de penalizacion,
em el documento pdf hablo mas a detalle sobre esto, de forma general y resumida es para realizar una comparacion de los diferentes modelos.

Se deben tener isntaladas las librerias de: 
NUMPY
PANDAS
SEABORN
MATPLOTLIB.PYPLOT
SKLEARN

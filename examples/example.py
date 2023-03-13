# Se importan los módulos necesarios para ejecutar el código, incluyendo
# TensorFlow para construir y entrenar el modelo, NumPy para trabajar con los
# datos y Matplotlib para visualizar los resultados.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento: se definen dos conjuntos de datos, "mi" y "km", que
# representan una relación entre millas e kilómetros, utilizados para entrenar
# el modelo.
mi = np.array([40, 20, 5.5, 1.3, 100, 1000, 40950], dtype=float)
km = np.array([64.37, 32.19, 8.85, 2.09, 160.93, 1609.34, 65902.64], dtype=float)

# Construcción del modelo: se define una capa densa con una única unidad, que
# se utiliza para construir un modelo secuencial que será entrenado con los
# datos. Además, se especifica el optimizador (Adam) y la función de
# pérdida (mean_squared_error) para el modelo.
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento del modelo: se entrena el modelo con los datos utilizando el
# método "fit" de Keras, especificando el número de épocas y si se desea
# mostrar información detallada sobre el proceso de entrenamiento (verbose).
print('Entrenando el modelo...')
history = model.fit(mi, km, epochs=1000, verbose=False)
print('¡Modelo entredano!')

# Visualización de los resultados: se utiliza Matplotlib para trazar una
# gráfica de la magnitud de la pérdida (loss) del modelo a lo largo de las
# épocas de entrenamiento. También se guarda la gráfica como una imagen PNG.
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de pérdida')
plt.plot(history.history['loss'])
plt.savefig('example.png', dpi=300, format='png')

# Interacción del usuario: se proporciona una interfaz simple para que el
# usuario pueda ingresar valores y obtener las predicciones del modelo
# correspondientes. El bucle "while" sigue pidiendo al usuario que ingrese un
# número hasta que escriba "q" para salir.
inp = str(input('>>> '))
while inp != 'q':
    number = float(inp)
    prediction = model.predict([number])
    print(f'Prection: {number} = {prediction}')
    inp = str(input('>>> '))

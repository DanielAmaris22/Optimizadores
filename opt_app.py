import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Título de la aplicación
st.title("API optimizadores")

with st.container():
    # Definir las dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.header("Limites de X")
        num1 = st.number_input("Limite Inferior x", value=0.0, step=0.01)
        num2 = st.number_input("Limite Superior x", value=0.0, step=0.01)
        
        st.header("Limites de Y")
        num3 = st.number_input("Limite Inferior y", value=0.0, step=0.01)
        num4 = st.number_input("Limite Superior y", value=0.0, step=0.01)
        
        if st.button("Graficar"):

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

            # Generar los datos.
            X = np.arange(num1, num2, 0.5)
            Y = np.arange(num3, num4, 0.5)
            X, Y = np.meshgrid(X, Y)    
            R = np.sqrt(X**2 + Y**2)
            Z = -np.sin(R)

            # Graficar la superficie.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

            # Personalizar el eje Z
            ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter('{x:.02f}')

            # Agregar una barra de colores para el mapeo de valores
            fig.colorbar(surf, shrink=0.5, aspect=5)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

    # ============================================
    # COLUMNA DERECHA: Triplicar un número
    # ============================================
    with col2:
        st.header("Metodos Optimizacion")
        metodo = st.selectbox("Seleccion metodo", ['Gradiente descendente', 'Gradiente descendente estocástico',
                                                   'RMSprop', 'Algoritmo Adam'])
        
        if metodo == 'Gradiente descendente':
            st.write("Parámetros para Gradiente descendente:")
            x_t = st.number_input("Ingrese el valor de theta[0]", value=0.0, step=0.01)
            y_t = st.number_input("Ingrese el valor de theta[1]", value=0.0, step=0.01)
            eta = st.number_input("Ingrese taza de aprendizaje", value=0.0, step=0.01)
            epochs = st.number_input("Numero de interacciones", value=0)

            theta_init = np.array([x_t, y_t])
            st.write(f"theta_init: {theta_init}")

        elif metodo == 'Gradiente descendente estocástico':
            st.write("Parámetros para Gradiente descendente estocástico:")
            x_t = st.number_input("Ingrese el valor de theta[0]", value=0.0, step=0.01)
            y_t = st.number_input("Ingrese el valor de theta[1]", value=0.0, step=0.01)
            eta = st.number_input("Ingrese taza de aprendizaje", value=0.0, step=0.01)
            epochs = st.number_input("Numero de interacciones", value=0)

            theta_init = np.array([x_t, y_t])
            st.write(f"theta_init: {theta_init}")
        
        elif metodo == 'RMSprop':
            st.write("Parámetros para RMSprop:")
            x_t = st.number_input("Ingrese el valor de theta[0]", value=0.0, step=0.01)
            y_t = st.number_input("Ingrese el valor de theta[1]", value=0.0, step=0.01)
            epochs = st.number_input("Numero de interacciones", value=0)

            theta_init = np.array([x_t, y_t])
            st.write(f"theta_init: {theta_init}")

        elif metodo == 'Algoritmo Adam':
            st.write("Parámetros para Adam:")
            x_t = st.number_input("Ingrese el valor de theta[0]", value=0.0, step=0.01)
            y_t = st.number_input("Ingrese el valor de theta[1]", value=0.0, step=0.01)
            epochs = st.number_input("Numero de interacciones", value=0)

            theta_init = np.array([x_t, y_t])
            st.write(f"theta_init: {theta_init}")


        if st.button("Calcular"):
            if metodo == 'Gradiente descendente':
                # Definir la función de pérdida
                def loss_func(theta):
                    x, y = theta
                    R = np.sqrt(x**2 + y**2)
                    return -np.sin(R)

                # Definir el gradiente de la función de pérdida
                def evaluate_gradient(loss_func, x_train, y_train, theta):
                    x, y = theta
                    R = np.sqrt(x**2 + y**2)
                    grad_x = -np.cos(R) * (x / R)
                    grad_y = -np.cos(R) * (y / R)
                    return np.array([grad_x, grad_y])

                # Gradiente descendente
                def gd(theta, x_train, y_train, loss_func, epochs, eta):
                    for i in range(epochs):
                        gradient = evaluate_gradient(loss_func, x_train, y_train, theta)
                        theta -= eta * gradient
                        # theta = theta - eta * gradient
                    return theta, gradient

                # Ejecutar gradiente descendente
                theta_final, gradient_final = gd(theta_init, None, None, loss_func, epochs, eta)
                st.write(f"El punto mínimo aproximado es: {theta_final}")
            
            elif metodo == 'Gradiente descendente estocástico':
                def loss_func(theta):
                    x, y = theta
                    R = np.sqrt(x**2 + y**2)
                    return -np.sin(R)

                # Definir el gradiente de la función de pérdida
                def evaluate_gradient(loss_func, x, y, theta):
                    R = np.sqrt(x**2 + y**2)
                    grad_x = -np.cos(R) * (x / R)
                    grad_y = -np.cos(R) * (y / R)
                    return np.array([grad_x, grad_y])

                # Gradiente descendente estocástico
                def sgd(theta, data_train, loss_func, epochs, eta):
                    for i in range(epochs):
                        np.random.shuffle(data_train)  # Barajar los datos en cada época
                        for example in data_train:
                            x, y = example
                            gradient = evaluate_gradient(loss_func, x, y, theta)
                            theta = theta - eta * gradient  # Actualizar los parámetros con el gradiente
                    return theta, gradient

                # Generar datos de entrenamiento (pueden ser puntos aleatorios en el plano)
                n_points = 100
                x_train = np.random.uniform(-6.5, 6.5, n_points)
                y_train = np.random.uniform(-6.5, 6.5, n_points)
                data_train = list(zip(x_train, y_train))  # Crear pares de datos (x, y)

                # Ejecutar el SGD
                theta_final, gradient_final = sgd(theta_init, data_train, loss_func, epochs, eta)
                st.write(f"El punto mínimo aproximado es: {theta_final}")

            elif metodo == 'RMSprop':
                def loss_func(theta):
                    x, y = theta
                    R = np.sqrt(x**2 + y**2)
                    return -np.sin(R)

                # Definir el gradiente de la función de pérdida
                def evaluate_gradient(loss_func, x, y, theta):
                    R = np.sqrt(x**2 + y**2)
                    grad_x = -np.cos(R) * (x / R)
                    grad_y = -np.cos(R) * (y / R)
                    return np.array([grad_x, grad_y])

                # RMSprop
                def rmsprop(theta, data_train, loss_func, epochs, eta=0.001, decay=0.9, epsilon=1e-8):
                    E_g2 = np.zeros_like(theta)  # Inicializar E[g^2] en cero
                    for epoch in range(epochs):
                        np.random.shuffle(data_train)  # Barajar los datos
                        for example in data_train:
                            x, y = example
                            gradient = evaluate_gradient(loss_func, x, y, theta)
            
                            # Actualizar el promedio del cuadrado del gradiente
                            E_g2 = decay * E_g2 + (1 - decay) * gradient**2
            
                            # Actualizar los parámetros usando RMSprop
                            theta -= eta / (np.sqrt(E_g2) + epsilon) * gradient
            
                    return theta

                # Generar datos de entrenamiento (pueden ser puntos aleatorios en el plano)
                n_points = 100
                x_train = np.random.uniform(-6.5, 6.5, n_points)
                y_train = np.random.uniform(-6.5, 6.5, n_points)
                data_train = list(zip(x_train, y_train))  # Crear pares de datos (x, y)

                # Ejecutar RMSprop
                theta_final = rmsprop(theta_init, data_train, loss_func, epochs)
                st.write(f"El punto mínimo aproximado es: {theta_final}")

            elif metodo == 'Algoritmo Adam':
                # Definir la función de pérdida
                def loss_func(theta):
                    x, y = theta
                    R = np.sqrt(x**2 + y**2)
                    return -np.sin(R)

                # Definir el gradiente de la función de pérdida
                def evaluate_gradient(loss_func, x, y, theta):
                    R = np.sqrt(x**2 + y**2)
                    grad_x = -np.cos(R) * (x / R)
                    grad_y = -np.cos(R) * (y / R)
                    return np.array([grad_x, grad_y])

                # Algoritmo Adam
                def adam(theta, data_train, loss_func, epochs, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
                    m = np.zeros_like(theta)  # Inicializar el momento de primer orden
                    v = np.zeros_like(theta)  # Inicializar el momento de segundo orden
                    t = 0  # Inicializar el contador de iteraciones

                    for epoch in range(epochs):
                        np.random.shuffle(data_train)  # Barajar los datos
                        for example in data_train:
                            x, y = example
                            t += 1  # Incrementar el contador
                            gradient = evaluate_gradient(loss_func, x, y, theta)

                            # Actualizar los momentos de primer y segundo orden
                            m = beta1 * m + (1 - beta1) * gradient
                            v = beta2 * v + (1 - beta2) * (gradient**2)

                            # Corrección de sesgo para momentos de primer y segundo orden
                            m_hat = m / (1 - beta1**t)
                            v_hat = v / (1 - beta2**t)

                            # Actualización de los parámetros
                            theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

                    return theta

                # Generar datos de entrenamiento (pueden ser puntos aleatorios en el plano)
                n_points = 100
                x_train = np.random.uniform(-6.5, 6.5, n_points)
                y_train = np.random.uniform(-6.5, 6.5, n_points)
                data_train = list(zip(x_train, y_train))  # Crear pares de datos (x, y)

                # Ejecutar Adam
                theta_final = adam(theta_init, data_train, loss_func, epochs)
                st.write(f"El punto mínimo aproximado es: {theta_final}")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    # Inicializa el entorno
    env = gym.make('CliffWalking-v0')

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.3            # Tasa de aprendizaje
    discount_factor = 0.99         # Factor de descuento para las recompensas
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.001     # Tasa de decaimiento de epsilon para reducir la exploración con el tiempo
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno cada 100 episodios, alternando entre modos con y sin renderización
        #? Para entrenar más rapido se puede comentar la siguiente línea de if
        if (i + 1) % 100 == 0:
            env.close()
            env = gym.make('CliffWalking-v0', render_mode='human')
        else:
            env.close()
            env = gym.make('CliffWalking-v0')

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Tomar una acción en base a si es explotación o exploración basado en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor acción basada en Q-table

            # Realizar la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q con la nueva información obtenida
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = reward

        # Imprime el progreso cada 10 episodios
        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para la inspección
    print(f"Mejor Q: {q_table}")

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 10 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100):(t + 1)])

    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas en los últimos 10 episodios')
    plt.title('Progreso del entrenamiento')
    plt.show()

# Ejecutar la función de entrenamiento si el script es el programa principal
if __name__ == '__main__':
    train(5000)

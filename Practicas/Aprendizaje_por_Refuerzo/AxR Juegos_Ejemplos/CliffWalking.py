import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train_cliff_walking(episodes):
    # Inicializa el entorno
    env = gym.make("CliffWalking-v0")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.1          # Tasa de aprendizaje
    discount_factor = 0.95       # Factor de descuento para las recompensas
    epsilon = 1                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.001   # Tasa de decaimiento de epsilon para reducir la exploración con el tiempo
    rng = np.random.default_rng() # Generador de números aleatorios

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]
        
        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Tomar una acción en base a si es exploración o explotación basado en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor basada en Q-table

            # Realizar la acción y obtener el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Actualiza la tabla Q con la nueva información obtenida
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])

            # Actualizar el estado para el siguiente paso
            state = new_state

            # Acumula la recompensa del episodio actual
            rewards_por_episode[i] += reward
        
        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa acumulada: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para la inspección
    print(f"Mejor Q: {q_table}")

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100):(t + 1)])
        
    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de Recompensas en Bloques de 100 Episodios')
    plt.title('Desempeño de Q-learning en CliffWalking-v0')
    plt.show()

# Ejecutar la función de entrenamiento si el script es el programa principal
if __name__ == '__main__':
    train_cliff_walking(20000)

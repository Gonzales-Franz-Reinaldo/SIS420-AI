import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def train(episodes, save_q_table=True, save_plot=True):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Contadores para la actualización incremental
    action_counts = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_sum = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.001     # Tasa de decaimiento de epsilon más rápida
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        
        # Reinicia el entorno cada 500 episodios, alternando entre modos con y sin renderización
        if (i + 1) % 500 == 0:
            env.close()
            env = gym.make("Taxi-v3", render_mode="human")
        else:
            env.close()
            env = gym.make("Taxi-v3")

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

            # Actualiza el contador de acciones y la suma de recompensas
            action_counts[state, action] += 1
            rewards_sum[state, action] += reward

            # Actualiza la tabla Q con la nueva información obtenida (método de acción-valor)
            q_table[state, action] = rewards_sum[state, action] / action_counts[state, action]

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = reward

        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para la inspección
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100) :(t + 1)])

    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    if save_plot:
        plt.savefig('recompensas.png')
    plt.show()

    # Guardar la tabla Q en un archivo pkl
    if save_q_table:
        with open('taxi.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        print("Tabla Q guardada en 'taxi.pkl'.")

def exploit(q_table_file='taxi.pkl', num_runs=10):
    # Cargar la tabla Q desde el archivo
    with open(q_table_file, 'rb') as f:
        q_table = pickle.load(f)
    print("Tabla Q cargada desde el archivo.")

    for run in range(num_runs):
        # Inicializa el entorno
        env = gym.make("Taxi-v3", render_mode="human")
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        # Exploitation loop
        while not terminated and not truncated:
            action = np.argmax(q_table[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            state = new_state
            env.render()
        
        print(f"Run {run + 1}: Recompensa total durante la explotación: {rewards}")
        env.close()

if __name__ == "__main__":
    # Cambia este valor a True para cargar el Q-table y explotar directamente
    exploit_directly = True

    if exploit_directly and os.path.exists('taxi.pkl'):
        exploit(num_runs=10)  # Ajusta el número de ejecuciones de explotación aquí
    else:
        train(10000)  # Aumenta el número de episodios para el entrenamiento

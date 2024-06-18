import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os

def train(episodes, save_q_table=True, save_plot=True):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    action_counts = np.ones((env.observation_space.n, env.action_space.n))  # Inicializa con unos para evitar división por cero

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.3
    discount_factor = 0.9
    c = 2.0  # Constante de exploración para UCB ajustada
    rng = np.random.default_rng()

    rewards_por_episode = np.zeros(episodes)

    for i in range(episodes):
        if (i + 1) % 100 == 0:
            env.close()
            env = gym.make("Taxi-v3", render_mode="human")
        else:
            env.close()
            env = gym.make("Taxi-v3")

        state = env.reset()[0]
        terminated = False
        truncated = False
        t = 0  # Contador de pasos dentro del episodio

        while not terminated and not truncated:
            t += 1

            # Implementación de UCB
            ucb_values = q_table[state, :] + c * np.sqrt(np.log(t + 1) / action_counts[state, :])
            action = np.argmax(ucb_values)

            new_state, reward, terminated, truncated, _ = env.step(action)
            action_counts[state, action] += 1

            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state

        rewards_por_episode[i] = reward

        if (i + 1) % 10 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    env.close()
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 10):(t + 1)])

    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    if save_plot:
        plt.savefig('recompensas_ucb.png')
    plt.show()

    # Guardar la tabla Q en un archivo pkl
    if save_q_table:
        with open('taxi_ucb.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        print("Tabla Q guardada en 'taxi_ucb.pkl'.")

def exploit(q_table_file='taxi_ucb.pkl', num_runs=10):
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

    if exploit_directly and os.path.exists('taxi_ucb.pkl'):
        exploit(num_runs=10)  # Ajusta el número de ejecuciones de explotación aquí
    else:
        train(5000)  # Ajusta el número de episodios para el entrenamiento

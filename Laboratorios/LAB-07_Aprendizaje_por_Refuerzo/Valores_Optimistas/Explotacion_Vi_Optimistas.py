import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False, optimistic_initial_value=10):
    # Inicializa el entorno Taxi-v3 de Gymnasium
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        # Inicializa la tabla Q con valores optimistas
        q = np.ones((env.observation_space.n, env.action_space.n)) * optimistic_initial_value
    else:
        # Carga la tabla Q desde un archivo
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    # Parámetros de aprendizaje
    learning_rate_a = 0.9  # Tasa de aprendizaje (alpha)
    discount_factor_g = 0.9  # Factor de descuento (gamma)
    epsilon = 1  # Probabilidad de tomar una acción aleatoria (100% al inicio)
    epsilon_decay_rate = 0.0001  # Tasa de decremento de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios

    rewards_per_episode = np.zeros(episodes)  # Para almacenar las recompensas por episodio

    for i in range(episodes):
        state = env.reset()[0]  # Reinicia el entorno y obtiene el estado inicial
        terminated = False  # Indica si el episodio ha terminado (llegó a destino)
        truncated = False  # Indica si se ha superado el límite de pasos

        rewards = 0  # Recompensas acumuladas en el episodio

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                # Acción aleatoria (exploración)
                action = env.action_space.sample()
            else:
                # Acción basada en la política actual (explotación)
                action = np.argmax(q[state, :])

            # Ejecuta la acción y obtiene el nuevo estado, recompensa, y si terminó/truncado
            new_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward  # Suma la recompensa obtenida

            if is_training:
                # Actualiza la tabla Q
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            state = new_state  # Actualiza el estado actual

        # Decaimiento de epsilon para reducir la exploración con el tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Ajusta la tasa de aprendizaje si epsilon llega a 0
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards  # Almacena la recompensa del episodio
        

    env.close()  # Cierra el entorno

    # Grafica la suma de recompensas de los últimos 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.savefig('taxi.png')

    # Guarda la tabla Q en un archivo
    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)
    
    # Mostrar la tabla Q final en consola
    print("Tabla Q final:")
    print(q)

if __name__ == '__main__':
    # run(15000, optimistic_initial_value=10)  # Entrena el agente con 15000 episodios con valores iniciales optimistas
    run(10, is_training=False, render=True)  # Ejecuta el agente entrenado con 10 episodios y renderizado

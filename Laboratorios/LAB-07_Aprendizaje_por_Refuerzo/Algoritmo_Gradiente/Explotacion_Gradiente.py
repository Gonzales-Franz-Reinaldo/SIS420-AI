import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards, dtype=float)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * discount_factor + rewards[i]
        discounted_rewards[i] = cumulative
    return discounted_rewards

def train(episodes, save_h_table=True, save_plot=True):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Crea la tabla de preferencias H inicializada con valores aleatorios pequeños
    H = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo de ascenso por gradiente
    learning_rate = 0.1  # Tasa de aprendizaje
    discount_factor = 0.9  # Factor de descuento
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Alternar renderización cada 1000 episodios
        render_mode = "human" if (i + 1) % 1000 == 0 else None
        if render_mode == "human":
            env.close()
            env = gym.make("Taxi-v3", render_mode=render_mode)
        else:
            env.reset()

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False

        # Variables para almacenar la recompensa acumulada en el episodio y las acciones realizadas
        episode_reward = 0
        rewards = []
        states = []
        actions = []

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Calcula las probabilidades de seleccionar cada acción
            action_probs = np.exp(H[state]) / np.sum(np.exp(H[state]))

            # Selecciona una acción basada en las probabilidades
            action = rng.choice(env.action_space.n, p=action_probs)

            # Realiza la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Almacena la recompensa, el estado y la acción
            rewards.append(reward)
            states.append(state)
            actions.append(action)

            # Acumula la recompensa del episodio
            episode_reward += reward

            # Actualiza el estado para el siguiente paso
            state = new_state

        # Calcular las recompensas descontadas
        discounted_rewards = discount_rewards(rewards, discount_factor)
        avg_reward = np.mean(discounted_rewards)

        # Actualiza las preferencias H utilizando el algoritmo de ascenso por gradiente
        for t in range(len(rewards)):
            state = states[t]
            action = actions[t]
            action_probs = np.exp(H[state]) / np.sum(np.exp(H[state]))
            for a in range(env.action_space.n):
                if a == action:
                    H[state, a] += learning_rate * (discounted_rewards[t] - avg_reward) * (1 - action_probs[a])
                else:
                    H[state, a] -= learning_rate * (discounted_rewards[t] - avg_reward) * action_probs[a]

        # Registra la recompensa obtenida en este episodio
        rewards_por_episode[i] = episode_reward

        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla H final para la inspección
    print("Preferencias de acción H resultantes después del entrenamiento:")
    print(H)

    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100):(t + 1)])
    
    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    if save_plot:
        plt.savefig('recompensas_gradiente.png')
    plt.show()

    # Guardar la tabla H en un archivo pkl
    if save_h_table:
        with open('taxi_gradiente.pkl', 'wb') as f:
            pickle.dump(H, f)
        print("Tabla H guardada en 'taxi_gradiente.pkl'.")

def exploit(h_table_file='taxi_gradiente.pkl', num_runs=10):
    # Cargar la tabla H desde el archivo
    with open(h_table_file, 'rb') as f:
        H = pickle.load(f)
    print("Tabla H cargada desde el archivo.")

    for run in range(num_runs):
        # Inicializa el entorno
        env = gym.make("Taxi-v3", render_mode="human")
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        # Exploitation loop
        while not terminated and not truncated:
            action_probs = np.exp(H[state]) / np.sum(np.exp(H[state]))
            action = np.argmax(action_probs)
            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            state = new_state
            env.render()
        
        print(f"Run {run + 1}: Recompensa total durante la explotación: {rewards}")
        env.close()

if __name__ == "__main__":
    # Cambia este valor a True para cargar el H-table y explotar directamente
    exploit_directly = True

    if exploit_directly:
        exploit(num_runs=10)  # Ajusta el número de ejecuciones de explotación aquí
    else:
        train(25000)  # Ajusta el número de episodios para el entrenamiento






















# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import os

# def discount_rewards(rewards, discount_factor):
#     discounted_rewards = np.zeros_like(rewards, dtype=float)
#     cumulative = 0.0
#     for i in reversed(range(len(rewards))):
#         cumulative = cumulative * discount_factor + rewards[i]
#         discounted_rewards[i] = cumulative
#     return discounted_rewards

# def run(episodes, is_training=True, render=False, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
#     # Inicializa el entorno Taxi-v3 de Gymnasium
#     env = gym.make('Taxi-v3', render_mode='human' if render else None)

#     if is_training:
#         # Inicializa la tabla de preferencias H con valores pequeños aleatorios
#         H = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))
#     else:
#         # Carga la tabla de preferencias H desde un archivo
#         with open(os.path.join(os.getcwd(), 'taxi.pkl'), 'rb') as f:
#             H = pickle.load(f)

#     rng = np.random.default_rng()  # Generador de números aleatorios
#     rewards_per_episode = np.zeros(episodes)  # Para almacenar las recompensas por episodio

#     for i in range(episodes):
#         state = env.reset()[0]  # Reinicia el entorno y obtiene el estado inicial
#         terminated = False  # Indica si el episodio ha terminado (llegó a destino)
#         truncated = False  # Indica si se ha superado el límite de pasos
#         episode_reward = 0  # Recompensas acumuladas en el episodio
#         rewards = []  # Lista de recompensas para el episodio
#         states = []  # Lista de estados para el episodio
#         actions = []  # Lista de acciones para el episodio

#         while not terminated and not truncated:
#             if rng.random() < epsilon:
#                 action = rng.choice(env.action_space.n)
#             else:
#                 # Normaliza H[state] para evitar overflow en la exponencial
#                 max_h = np.max(H[state])
#                 exp_h = np.exp(H[state] - max_h)
#                 action_probs = exp_h / np.sum(exp_h)
#                 action = rng.choice(env.action_space.n, p=action_probs)

#             # Ejecuta la acción y obtiene el nuevo estado, recompensa, y si terminó/truncado
#             new_state, reward, terminated, truncated, _ = env.step(action)

#             # Almacena las recompensas, estados y acciones
#             rewards.append(reward)
#             states.append(state)
#             actions.append(action)

#             episode_reward += reward  # Suma la recompensa obtenida
#             state = new_state  # Actualiza el estado actual

#         rewards_per_episode[i] = episode_reward  # Almacena la recompensa del episodio

#         if is_training:
#             # Calcular las recompensas descontadas
#             discounted_rewards = discount_rewards(rewards, discount_factor)
#             avg_reward = np.mean(discounted_rewards)

#             # Actualiza la tabla de preferencias H utilizando el algoritmo de ascenso por gradiente
#             for t in range(len(rewards)):
#                 state = states[t]
#                 action = actions[t]
#                 # Normaliza H[state] para evitar overflow en la exponencial
#                 max_h = np.max(H[state])
#                 exp_h = np.exp(H[state] - max_h)
#                 action_probs = exp_h / np.sum(exp_h)
#                 for a in range(env.action_space.n):
#                     if a == action:
#                         H[state, a] += learning_rate * (discounted_rewards[t] - avg_reward) * (1 - action_probs[a])
#                     else:
#                         H[state, a] -= learning_rate * (discounted_rewards[t] - avg_reward) * action_probs[a]

#     env.close()  # Cierra el entorno

#     # Grafica la suma de recompensas de los últimos 100 episodios
#     sum_rewards = np.zeros(episodes)
#     for t in range(episodes):
#         sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
#     plt.plot(sum_rewards)
#     plt.xlabel('Episodios')
#     plt.ylabel('Suma de recompensas acumuladas')
#     plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
#     plt.savefig(os.path.join(os.getcwd(), 'taxi.png'))

#     # Guarda la tabla de preferencias H en un archivo
#     if is_training:
#         with open(os.path.join(os.getcwd(), "taxi.pkl"), "wb") as f:
#             pickle.dump(H, f)
    
#     # Mostrar la tabla de preferencias H final en consola
#     print("Tabla de preferencias H final:")
#     print(H)

# if __name__ == '__main__':
#     run(20000)  # Entrena el agente con 1700 episodios
#     run(10, is_training=False, render=True)  # Ejecuta el agente entrenado con 10 episodios y renderizado

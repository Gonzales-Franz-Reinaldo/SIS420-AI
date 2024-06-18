import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards, dtype=float)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * discount_factor + rewards[i]
        discounted_rewards[i] = cumulative
    return discounted_rewards

def train(episodes):
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
        # Alternar renderización cada 100 episodios
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

        # Imprime el progreso cada 500 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {reward}")

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
    plt.show()

if __name__ == "__main__":
    train(20000)

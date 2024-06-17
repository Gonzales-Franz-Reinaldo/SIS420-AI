import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math

def train(episodes):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    action_counts = np.zeros((env.observation_space.n, env.action_space.n))  # Contador de acciones

    # Define los parámetros del algoritmo Q-learning
    learning_rate = 0.3
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0003
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

        while not terminated and not truncated:
            # Implementación de UCB
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                total_counts = np.sum(action_counts[state, :])
                if total_counts == 0:
                    action = env.action_space.sample()
                else:
                    confidence_bounds = q_table[state, :] + np.sqrt(2 * np.log(total_counts) / (action_counts[state, :] + 1e-10))
                    action = np.argmax(confidence_bounds)

            new_state, reward, terminated, truncated, _ = env.step(action)
            action_counts[state, action] += 1

            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0.01)
        rewards_por_episode[i] = reward

        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")

    env.close()
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100) :(t + 1)])

    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

if __name__ == "__main__":
    train(2500)

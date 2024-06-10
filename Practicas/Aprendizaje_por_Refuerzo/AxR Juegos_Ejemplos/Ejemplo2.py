import gym
import numpy as np
import random

# Parámetros
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Crear bins
state_bins = [
    np.linspace(-4.8, 4.8, 10),    # Cart position
    np.linspace(-4, 4, 10),        # Cart velocity
    np.linspace(-0.418, 0.418, 10),# Pole angle
    np.linspace(-4, 4, 10)         # Pole velocity at tip
]

# Definir forma de la tabla Q
q_table_shape = tuple(len(bins) + 1 for bins in state_bins) + (action_size,)
q_table = np.zeros(q_table_shape)

# Hiperparámetros
learning_rate = 0.1
discount_rate = 0.99
episodes = 1000
max_steps = 200
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Función de estado discreto
def discretize_state(state, bins):
    discrete_state = []
    for i in range(len(state)):
        discrete_state.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discrete_state)

# Función de depuración del estado
def debug_state(state, context):
    print(f"{context} - Tipo de estado: {type(state)}, Estado: {state}")

# Entrenamiento
for episode in range(episodes):
    state, _ = env.reset()  # Extraer solo el estado
    debug_state(state, "Estado inicial (reset)")
    state = np.array(state, dtype=np.float32)  # Asegurarse de que el estado sea un array de numpy
    debug_state(state, "Estado inicial (convertido a array)")
    state = discretize_state(state, state_bins)
    done = False
    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Acción aleatoria
        else:
            action = np.argmax(q_table[state])  # Acción de la tabla Q
        next_state, reward, done, _, _ = env.step(action)
        debug_state(next_state, "Próximo estado")
        next_state = np.array(next_state, dtype=np.float32)  # Asegurarse de que el próximo estado sea un array de numpy
        debug_state(next_state, "Próximo estado (convertido a array)")
        next_state = discretize_state(next_state, state_bins)
        if done:
            reward = -100
        q_table[state][action] += learning_rate * (reward + discount_rate * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state
        if done:
            break
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    if episode % 100 == 0:
        print(f"Episodio: {episode}, Epsilon: {epsilon}")

# Evaluación del agente
for episode in range(5):
    state, _ = env.reset()  # Extraer solo el estado
    state = np.array(state, dtype=np.float32)  # Asegurarse de que el estado sea un array de numpy
    state = discretize_state(state, state_bins)
    done = False
    total_reward = 0
    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)  # Asegurarse de que el próximo estado sea un array de numpy
        next_state = discretize_state(next_state, state_bins)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Recompensa total en el episodio {episode}: {total_reward}")
env.close()

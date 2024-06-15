import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt

# Crear la red neuronal para DQN
def create_q_model(input_shape, action_space):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    return model

# Clase del agente DQN
class DQNAgent:
    def __init__(self, state_shape, action_space, batch_size=32):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.batch_size = batch_size
        self.model = create_q_model(state_shape, action_space)
        self.target_model = create_q_model(state_shape, action_space)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.loss_function = tf.keras.losses.Huber()
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Preprocesamiento de las observaciones
def preprocess_frame(frame):
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.resize(frame, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    frame = tf.cast(frame, tf.float32) / 255.0
    return frame

def stack_frames(frames, new_frame, is_new_episode):
    new_frame = preprocess_frame(new_frame)
    if is_new_episode:
        frames = np.stack([new_frame] * 4, axis=2)
    else:
        frames = np.append(new_frame, frames[:, :, :3], axis=2)
    return frames

# Entrenamiento del agente
def train_agent(env, episodes):
    action_space = env.action_space.n
    state_shape = (84, 84, 4)
    agent = DQNAgent(state_shape, action_space)
    rewards = []

    for ep in range(episodes):
        state = env.reset()[0]
        state = stack_frames(None, state, True)
        total_reward = 0
        done = False

        while not done:
            env.render()
            action = agent.act(np.expand_dims(state, axis=0))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = stack_frames(state, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        agent.update_target_network()
        rewards.append(total_reward)
        print(f"Episodio {ep + 1}: Recompensa total: {total_reward}")

    plt.plot(rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title('Desempeño de DQN en IceHockey-v5')
    plt.show()

# Ejecutar la función de entrenamiento si el script es el programa principal
if __name__ == '__main__':
    env = gym.make("ALE/IceHockey-v5", render_mode='human')
    train_agent(env, 1000)

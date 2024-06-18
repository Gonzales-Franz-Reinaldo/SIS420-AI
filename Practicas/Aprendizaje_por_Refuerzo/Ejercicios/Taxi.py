import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

def guardarQ_table(qtable, filename='taxi.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(qtable, f)
    print(f"Q-table saved to {filename}")

def cargarQ_table(filename='taxi.pkl'):
    with open(filename, 'rb') as f:
        qtable = pickle.load(f)
    print(f"Q-table loaded from {filename}")
    return qtable

def plot_rewards(rewards):
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Episodes')
    plt.show()

def main(train=True):
    env = gym.make("Taxi-v3", render_mode="human")

    state_size = env.observation_space.n
    action_size = env.action_space.n

    if train:
        qtable = np.zeros((state_size, action_size))

        EPISODES = 1000
        STEPS_PER_EPISODE = 100

        epsilon = 1.0
        decay_rate = 0.005
        learning_rate = 0.9
        discount_rate = 0.8

        rewards = []

        for episode in range(EPISODES):
            done = False
            state = env.reset()[0]
            total_rewards = 0

            for step in range(STEPS_PER_EPISODE):
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qtable[state, :])

                new_state, reward, done, truncated, info = env.step(action)
                qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

                state = new_state
                total_rewards += reward

                if done or truncated:
                    break

            epsilon = np.exp(-decay_rate * episode)
            rewards.append(total_rewards)

        plot_rewards(rewards)
        guardarQ_table(qtable)

    else:
        qtable = cargarQ_table()

    # Test the trained agent
    state = env.reset()[0]
    done = False
    rewards = 0

    for s in range(STEPS_PER_EPISODE):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])
        new_state, reward, done, truncated, info = env.step(action)
        rewards += reward
        env.render()

        print(f"score: {rewards}")
        state = new_state

        if done or truncated:
            break

    env.close()

if __name__ == "__main__":
    # para probar el modelo True para entrenar y FALSE para cargar el modelo entrenado
    main(train=True) 

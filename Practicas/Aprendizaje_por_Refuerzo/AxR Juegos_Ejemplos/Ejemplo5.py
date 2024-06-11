import gym
import random
import numpy as np


def main():
    env = gym.make("Taxi-v3")

    # Initialize the q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Set the number of episodes
    EPISODES = 1000
    STEPS_PER_EPISODE = 99

    # Hyperparameters
    epsilon = 1.0
    decay_rate = 0.005
    learning_rate = 0.9
    discount_rate = 0.8

    for episode in range(EPISODES):
        done = False
        state = env.reset()

        for step in range(STEPS_PER_EPISODE):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            state = new_state

            if done or truncated:
                break

        epsilon = np.exp(-decay_rate * episode)

    state = env.reset()
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
    main()
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_q_table(Q_table, states):
    plt.figure(figsize=(12, 8))
    for state in states:
        ax = sns.heatmap(Q_table[state].reshape(1, -1), annot=True, cmap="YlGnBu", cbar=False, linewidths=.5, linecolor='black')
        ax.set_title(f"State {state}")
        ax.set_yticklabels(['Actions'])
        ax.set_xticklabels(['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'])
        plt.show()


def main():
    env = gym.make("Taxi-v3", render_mode="human")

    # Initialize the q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q_table = np.zeros((state_size, action_size))

    # Set the number of episodes
    EPISODES = 1000
    STEPS_PER_EPISODE = 200  # Adjusted according to documentation

    # Hyperparameters
    epsilon = 1.0
    decay_rate = 0.005
    learning_rate = 0.9
    discount_rate = 0.8

    for episode in range(EPISODES):
        state = env.reset()
        total_rewards = 0

        if isinstance(state, tuple):
            state = state[0]

        for step in range(STEPS_PER_EPISODE):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state, :])

            if not isinstance(action, int):
                action = int(action)

            result = env.step(action)
            if len(result) == 4:
                new_state, reward, done, info = result
                truncated = False
            else:
                new_state, reward, done, truncated, info = result

            if isinstance(new_state, tuple):
                new_state = new_state[0]

            # Metodo Imcremental
            # Formula para calcular la Q para los valores de los estados 
            Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + discount_rate * np.max(Q_table[new_state, :]) - Q_table[state, action])

            state = new_state
            total_rewards += reward

            # Print the Q-table update
            print(f"Episode: {episode}, Step: {step}")
            print(f"State: {state}, Action: {action}, Reward: {reward}")
            print("Q-table:")
            print(Q_table)

            if done or truncated:
                break

        # Print total rewards for this episode
        print(f"Episode: {episode} finished with total reward: {total_rewards}")

        # Reduce epsilon (exploration rate)
        epsilon = np.exp(-decay_rate * episode)
        print(f"Epsilon after episode {episode}: {epsilon}")

    # Print the final Q-table
    print("Final Q-table:")
    print(Q_table)

    # Plotting the Q-table for selected states
    example_states = [0, 10, 50, 100, 200]
    plot_q_table(Q_table, example_states)

    # Run the trained agent
    state = env.reset()
    done = False
    rewards = 0

    if isinstance(state, tuple):
        state = state[0]

    for s in range(STEPS_PER_EPISODE):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(Q_table[state, :])

        if not isinstance(action, int):
            action = int(action)

        result = env.step(action)
        if len(result) == 4:
            new_state, reward, done, info = result
            truncated = False
        else:
            new_state, reward, done, truncated, info = result

        if isinstance(new_state, tuple):
            new_state = new_state[0]

        rewards += reward
        env.render()

        print(f"score: {rewards}")
        state = new_state

        if done or truncated:
            break

    env.close()


# Función principal para ejecutar el modelo 
if __name__ == "__main__":
    main()
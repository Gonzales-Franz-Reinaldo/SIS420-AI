import gym
import random
import numpy as np

def main():
    # Crear el entorno de Taxi-v3
    env = gym.make("Taxi-v3", render_mode="human")

    # Inicializar la tabla Q con ceros
    state_size = env.observation_space.n  # Número de estados posibles (500)
    action_size = env.action_space.n  # Número de acciones posibles (6)
    qtable = np.zeros((state_size, action_size))

    # Definir el número de episodios y pasos por episodio
    EPISODES = 1000
    STEPS_PER_EPISODE = 200  # Ajustado según la documentación

    # Hiperparámetros
    epsilon = 1.0  # Tasa de exploración inicial
    decay_rate = 0.005  # Tasa de decaimiento de epsilon
    alpha = 0.5  # Tasa de aprendizaje
    discount_rate = 0.8  # Factor de descuento

    for episode in range(EPISODES):
        state = env.reset()  # Reiniciar el entorno al inicio de cada episodio
        total_rewards = 0  # Inicializar la recompensa total por episodio

        if isinstance(state, tuple):
            state = state[0]  # Si el estado es una tupla, tomar el primer elemento

        for step in range(STEPS_PER_EPISODE):
            # Elegir una acción: exploración o explotación
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Acción aleatoria (exploración)
            else:
                action = np.argmax(qtable[state, :])  # Mejor acción basada en la tabla Q (explotación)

            if not isinstance(action, int):
                action = int(action)  # Asegurarse de que la acción sea un entero

            result = env.step(action)  # Ejecutar la acción y obtener el resultado
            if len(result) == 4:
                new_state, reward, done, info = result  # Desempaquetar resultado (v0.25.0+)
                truncated = False
            else:
                new_state, reward, done, truncated, info = result  # Desempaquetar resultado (anterior)

            if isinstance(new_state, tuple):
                new_state = new_state[0]  # Si el nuevo estado es una tupla, tomar el primer elemento

            # Actualizar la tabla Q usando la fórmula incremental
            qtable[state, action] += alpha * (reward - qtable[state, action])

            state = new_state  # Actualizar el estado actual
            total_rewards += reward  # Acumular la recompensa

            # Imprimir la actualización de la tabla Q
            print(f"Episode: {episode}, Step: {step}")
            print(f"State: {state}, Action: {action}, Reward: {reward}")
            print("Q-table:")
            print(qtable)

            if done or truncated:
                break  # Terminar el episodio si está completado o truncado

        # Imprimir las recompensas totales para este episodio
        print(f"Episode: {episode} finished with total reward: {total_rewards}")

        # Reducir epsilon (tasa de exploración)
        epsilon = np.exp(-decay_rate * episode)
        print(f"Epsilon after episode {episode}: {epsilon}")

    # Imprimir la tabla Q final
    print("Final Q-table:")
    print(qtable)

    # Ejecutar el agente entrenado
    state = env.reset()
    done = False
    rewards = 0

    if isinstance(state, tuple):
        state = state[0]  # Si el estado es una tupla, tomar el primer elemento

    for s in range(STEPS_PER_EPISODE):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])  # Elegir la mejor acción según la tabla Q

        if not isinstance(action, int):
            action = int(action)  # Asegurarse de que la acción sea un entero

        result = env.step(action)  # Ejecutar la acción
        if len(result) == 4:
            new_state, reward, done, info = result  # Desempaquetar resultado (v0.25.0+)
            truncated = False
        else:
            new_state, reward, done, truncated, info = result  # Desempaquetar resultado (anterior)

        if isinstance(new_state, tuple):
            new_state = new_state[0]  # Si el nuevo estado es una tupla, tomar el primer elemento

        rewards += reward  # Acumular la recompensa
        env.render()  # Renderizar el entorno

        print(f"score: {rewards}")
        state = new_state  # Actualizar el estado actual

        if done or truncated:
            break  # Terminar si el episodio está completado o truncado

    env.close()  # Cerrar el entorno

if __name__ == "__main__":
    main()

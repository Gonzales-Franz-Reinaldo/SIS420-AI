import gym
import random
import math
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def main():
    # Crear el entorno de Taxi-v3
    env = gym.make("Taxi-v3", render_mode="human")

    # Inicializar las preferencias (H) y la matriz de recompensas medias
    state_size = env.observation_space.n  # Número de estados posibles (500)
    action_size = env.action_space.n  # Número de acciones posibles (6)
    H = np.zeros((state_size, action_size))  # Inicializar preferencias
    recompensas_medias = np.zeros((state_size, action_size))

    # Definir el número de episodios y pasos por episodio
    EPISODES = 1000
    STEPS_PER_EPISODE = 200  # Ajustado según la documentación

    # Hiperparámetros
    alpha = 0.1  # Tasa de aprendizaje

    for episode in range(EPISODES):
        state = env.reset()  # Reiniciar el entorno al inicio de cada episodio
        recompensas = []  # Lista para almacenar recompensas del episodio

        if isinstance(state, tuple):
            state = state[0]  # Si el estado es una tupla, tomar el primer elemento

        for step in range(STEPS_PER_EPISODE):
            # Calcular las probabilidades usando softmax
            pi = softmax(H[state, :])

            # Elegir una acción basada en las probabilidades
            action = np.random.choice(action_size, p=pi)

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

            recompensas.append(reward)  # Almacenar recompensa
            recompensa_media = np.mean(recompensas)  # Calcular recompensa media

            # Actualizar preferencias
            for j in range(action_size):
                if j == action:
                    H[state, j] += alpha * (reward - recompensa_media) * (1 - pi[j])
                else:
                    H[state, j] -= alpha * (reward - recompensa_media) * pi[j]

            state = new_state  # Actualizar el estado actual

            # Imprimir la actualización de las preferencias
            print(f"Episode: {episode}, Step: {step}")
            print(f"State: {state}, Action: {action}, Reward: {reward}")
            print("Preferences (H):")
            print(H)

            if done or truncated:
                break  # Terminar el episodio si está completado o truncado

        # Imprimir las recompensas totales para este episodio
        print(f"Episode: {episode} finished with total reward: {sum(recompensas)}")

    # Imprimir las preferencias finales
    print("Final Preferences (H):")
    print(H)

    # Ejecutar el agente entrenado
    state = env.reset()
    done = False
    rewards = 0

    if isinstance(state, tuple):
        state = state[0]  # Si el estado es una tupla, tomar el primer elemento

    for s in range(STEPS_PER_EPISODE):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        # Calcular las probabilidades usando softmax
        pi = softmax(H[state, :])
        action = np.random.choice(action_size, p=pi)  # Elegir acción basada en probabilidades

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

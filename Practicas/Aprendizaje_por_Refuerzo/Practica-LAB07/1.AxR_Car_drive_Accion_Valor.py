import gym
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
    STEPS_PER_EPISODE = 200  # Ajustado según la 

    # Hiperparámetros
    epsilons = [0, 0.01, 0.1]
    learning_rate = 0.9  # Tasa de aprendizaje
    discount_rate = 0.8  # Factor de descuento

    recompensas_medias = np.zeros((len(epsilons), STEPS_PER_EPISODE))
    acciones_optimas = np.zeros((len(epsilons), STEPS_PER_EPISODE))

    for ej in range(EPISODES):
        for i, epsilon in enumerate(epsilons):
            state = env.reset()  # Reiniciar el entorno al inicio de cada episodio
            total_rewards = 0  # Inicializar la recompensa total por episodio

            if isinstance(state, tuple):
                state = state[0]  # Si el estado es una tupla, tomar el primer elemento

            Q = np.zeros(action_size)  # Inicializar el diccionario de valores Q para las acciones
            acciones = np.zeros(action_size)  # Contador de acciones
            recompensas = np.zeros(action_size)  # Acumulador de recompensas

            for step in range(STEPS_PER_EPISODE):
                # Elegir una acción: exploración o explotación
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # Acción aleatoria (exploración)
                else:
                    maxQ = np.max(Q)
                    action = np.argmax(Q == maxQ)

                result = env.step(action)  # Ejecutar la acción y obtener el resultado
                if len(result) == 4:
                    new_state, reward, done, info = result  # Desempaquetar resultado (v0.25.0+)
                    truncated = False
                else:
                    new_state, reward, done, truncated, info = result  # Desempaquetar resultado (anterior)

                if isinstance(new_state, tuple):
                    new_state = new_state[0]  # Si el nuevo estado es una tupla, tomar el primer elemento

                acciones[action] += 1
                recompensas[action] += reward
                Q[action] = recompensas[action] / acciones[action]
                state = new_state  # Actualizar el estado actual
                total_rewards += reward  # Acumular la recompensa

                recompensas_medias[i][step] += reward
                acciones_optimas[i][step] += (action == np.argmax(qtable[state, :]))

                if done or truncated:
                    break  # Terminar el episodio si está completado o truncado

            # Imprimir las recompensas totales para este episodio
            print(f"Episode: {ej}, Epsilon: {epsilon}, finished with total reward: {total_rewards}")

    recompensas_medias /= EPISODES
    acciones_optimas /= EPISODES

    # Imprimir las recompensas medias y las acciones óptimas
    print("Recompensas medias:")
    print(recompensas_medias)
    print("Acciones óptimas:")
    print(acciones_optimas)

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


# Imprimimos las recompensas medias y las acciones óptimas:
if __name__ == "__main__":
    main()

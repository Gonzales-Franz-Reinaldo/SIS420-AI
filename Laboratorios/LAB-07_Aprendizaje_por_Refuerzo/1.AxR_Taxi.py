
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def accionPorValor(episodes):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    action_counts = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_sum = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.001    # Tasa de decaimiento de epsilon reducida para exploración más prolongada
    min_epsilon = 0.01             # Valor mínimo de epsilon
    rng = np.random.default_rng()  # Generador de números aleatorios
    
    # Variables para almacenar las recompensas medias y acciones óptimas por epsilon
    epsilons = [0.1, 0.7, 1]
    recompensas_medias = np.zeros((len(epsilons), episodes))
    acciones_optimas = np.zeros((len(epsilons), episodes))

    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno cada 50 episodios, alternando entre modos con y sin renderización
        if (i + 1) % 1000 == 0:
            env.close()
            env = gym.make("Taxi-v3", render_mode="human")
        else:
            env.close()
            env = gym.make("Taxi-v3")

        # Reinicia el entorno y obtiene el estado inicial
        state = env.reset()[0]

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False
        episode_rewards = 0

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Tomar una acción en base a si es explotación o exploración basado en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor acción basada en Q-table

            # Realizar la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)
            #! episode_rewards += reward

            # Actualiza el conteo de la acción
            action_counts[state, action] += 1

            # Actualiza la suma de recompensas para la acción
            rewards_sum[state, action] += reward

            # Actualiza la tabla Q con la nueva información obtenida (promedio)
            q_table[state, action] = rewards_sum[state, action] / action_counts[state, action]

            # Actualizar el estado para el siguiente paso
            state = new_state

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        #! # Registra la recompensa obtenida en este episodio
        # #! rewards_por_episode[i] = episode_rewards
        # rewards_por_episode[i] = reward

        
        # Actualiza las recompensas medias y acciones óptimas por epsilon
        for j, e in enumerate(epsilons):
            if np.random.uniform(0, 1) < e:
                a = np.random.randint(env.action_space.n)
            else:
                a = np.argmax(q_table[state, :])
            recompensa = q_table[state, a]
            recompensas_medias[j][i] += recompensa
            acciones_optimas[j][i] += (a == np.argmax(q_table[state, :]))


        # Imprime el progreso cada 50 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {reward}")
            
    # Calcula las medias de recompensas y acciones óptimas
    recompensas_medias /= episodes
    acciones_optimas /= episodes
    
    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprime la tabla Q final para la inspección
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Gráfica de la evolución de las recompensas medias por episodio para cada epsilon
    plt.figure(figsize=(8, 5))
    for i, e in enumerate(epsilons):
        plt.plot(recompensas_medias[i], label=f'$\epsilon$ = {e}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa media')
    plt.title('Evolución de las recompensas medias por episodio para cada epsilon')
    plt.show()

if __name__ == "__main__":
    accionPorValor(20000)


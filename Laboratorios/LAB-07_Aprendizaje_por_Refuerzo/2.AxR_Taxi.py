import gym
import numpy as np
import matplotlib.pyplot as plt

def train_incremental_q_learning(episodes):
    # Inicializa el entorno
    env = gym.make("Taxi-v3")

    # Crea la tabla Q inicializada con ceros para todas las combinaciones estado-acción
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parámetros del algoritmo Q-learning
    alpha = 0.1                    # Tasa de aprendizaje reducida
    discount_factor = 0.9          # Factor de descuento
    epsilon = 1.0                  # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.0003    # Tasa de decaimiento de epsilon reducida para exploración más prolongada
    rng = np.random.default_rng()  # Generador de números aleatorios

    # Variables para almacenar las recompensas medias y acciones óptimas por epsilon
    epsilons = [0.1, 0.5, 1.0]
    recompensas_medias = np.zeros((len(epsilons), episodes))
    acciones_optimas = np.zeros((len(epsilons), episodes))

    # Bucle principal de entrenamiento
    for i in range(episodes):
        # Reinicia el entorno cada 50 episodios, alternando entre modos con y sin renderización
        if (i + 1) % 50 == 0:
            env.close()
            env = gym.make("Taxi-v3", render_mode="human")
        else:
            env.close()
            env = gym.make("Taxi-v3")

        # Reinicia el entorno y obtiene el estado inicial
        state, _ = env.reset()

        # Variables para controlar la finalización del episodio
        terminated = False
        truncated = False
        total_reward = 0  # Para acumular la recompensa total del episodio

        # Bucle para cada paso dentro de un episodio
        while not terminated and not truncated:
            # Tomar una acción en base a si es explotación o exploración basado en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploración: Selecciona una acción aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotación: Selecciona la mejor acción basada en Q-table

            # Realizar la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Calcula la actualización de la tabla Q (método incremental)
            old_q_value = q_table[state, action]
            new_q_value = old_q_value + alpha * (reward - old_q_value)
            q_table[state, action] = new_q_value

            # Actualizar el estado para el siguiente paso
            state = new_state   

            # Acumular la recompensa total
            #! total_reward += reward

        # Imprimir el progreso cada 50 episodios
        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa total: {reward}")

        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        # Actualiza las recompensas medias y acciones óptimas por epsilon
        for j, e in enumerate(epsilons):
            Q = {k: 0 for k in range(6)}  # Claves de 0 a 5 para cubrir todas las posibles acciones
            for exp in range(episodes):
                # Elegir acción
                if np.random.uniform(0, 1) < e:
                    # Acción aleatoria
                    a = np.random.randint(6)  # Acciones de 0 a 5
                else:
                    # Acción con mayor valor
                    a = np.argmax(q_table[state, :])
                recompensa = q_table[state, a]
                Q[a] += alpha * (recompensa - Q[a])
                recompensas_medias[j][exp] += recompensa
                acciones_optimas[j][exp] += (a == np.argmax(q_table[state, :]))

    # Calcula las medias de recompensas y acciones óptimas
    recompensas_medias /= episodes
    acciones_optimas /= episodes

    # Cierra el entorno al finalizar el entrenamiento
    env.close()

    # Imprimir la tabla Q final
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Gráfica de la evolución de las recompensas medias por episodio para cada epsilon
    plt.figure(figsize=(8, 5))
    for i, e in enumerate(epsilons):
        plt.plot(recompensas_medias[i], label=f'$\\epsilon = {e}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa media')
    plt.title('Evolución de las recompensas medias por episodio para cada epsilon')

    plt.show()

if __name__ == "__main__":
    train_incremental_q_learning(2500)
    
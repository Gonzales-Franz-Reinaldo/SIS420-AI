import gym
import random
import numpy as np
import matplotlib.pyplot as plt  # Importar matplotlib para la visualización

def main():
    env = gym.make("Taxi-v3", render_mode="human")  # Especificar el modo de renderizado para la visualización

    # Inicializar la tabla Q
    state_size = env.observation_space.n  # Número de estados en el entorno
    action_size = env.action_space.n  # Número de acciones posibles en el entorno
    Q_table = np.zeros((state_size, action_size))  # Crear la tabla Q con ceros

    # Configurar el número de episodios
    EPISODES = 1000           # Número de episodios para entrenar
    STEPS_PER_EPISODE = 100   # Número máximo de pasos por episodio

    # Hiperparámetros
    epsilon = 1.0  # Tasa de exploración inicial
    decay_rate = 0.005  # Tasa de decaimiento de epsilon
    learning_rate = 0.9  # Tasa de aprendizaje
    discount_rate = 0.8  # Factor de descuento

    # Lista para almacenar las recompensas totales de cada episodio
    rewards_per_episode = []

    for episode in range(EPISODES):
        # Al comienzo de cada episodio, done es falso
        done = False
        # Reiniciar el entorno para cada nuevo episodio
        state = env.reset()[0]
        total_rewards = 0  # Inicializar las recompensas totales para el episodio

        print(f"Inicio del episodio {episode + 1}")

        for step in range(STEPS_PER_EPISODE):
            # Decidir si explorar el entorno o explotar el conocimiento actual
            if random.uniform(0, 1) < epsilon:
                # Explorar: seleccionar una acción aleatoria
                action = env.action_space.sample()
            else:
                # Explotar: seleccionar la mejor acción conocida
                action = np.argmax(Q_table[state, :])

            # Ejecutar la acción y obtener el nuevo estado y la recompensa
            new_state, reward, done, truncated, info = env.step(action)

            # Implementación del algoritmo Q-learning
            Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + discount_rate * np.max(Q_table[new_state, :]) - Q_table[state, action])

            # Actualizar el estado actual
            state = new_state

            # Acumular la recompensa total
            total_rewards += reward

            # Mostrar información del paso actual
            print(f"Episodio: {episode + 1}, Paso: {step + 1}, Estado: {state}, Accion: {action}, Recompensa: {reward}, Recompensas Totales: {total_rewards}")

            # Si el episodio ha terminado o se ha truncado, salir del bucle
            if done or truncated:
                break

        # Almacenar la recompensa total de este episodio
        rewards_per_episode.append(total_rewards)

        # Disminuir epsilon (tasa de exploración) exponencialmente
        epsilon = np.exp(-decay_rate * episode)

    # Reiniciar el entorno para ver cómo se comporta el agente entrenado
    state = env.reset()[0]
    done = False
    rewards = 0

    # Este bucle es para la animación, para que se pueda ver visualmente cómo se comporta el agente
    for s in range(STEPS_PER_EPISODE):
        print(f"AGENTE ENTRENADO")
        print("Paso {}".format(s + 1))

        # Explotar una acción conocida, solo usaremos la explotación ya que el agente ya está entrenado
        action = np.argmax(Q_table[state, :])
        
        # Ejecutar la acción en el entorno
        new_state, reward, done, truncated, info = env.step(action)
        
        # Actualizar la recompensa
        rewards += reward
        
        # Actualizar la visualización del entorno
        env.render()

        print(f"Puntuación: {rewards}")
        state = new_state

        # Si el episodio ha terminado o se ha truncado, salir del bucle
        if done or truncated:
            break

    # Imprimir la tabla Q resultante
    print("Tabla Q resultante después del entrenamiento:")
    print(Q_table)

    # Graficar la evolución de las recompensas a lo largo del tiempo
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas')
    plt.title('Evolución de las Recompensas por Episodio')
    plt.show()

    # Cerrar el entorno
    env.close()


if __name__ == "__main__":
    main()

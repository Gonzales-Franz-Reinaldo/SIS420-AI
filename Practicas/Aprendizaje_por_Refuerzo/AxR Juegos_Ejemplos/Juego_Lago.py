import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt

def train (episodes):
    #Inicializa el entorno 
    env = gym.make("FrozenLake-v1")
    
    #crea la tabla Q inicializada con ceros para todas las combinaciones estado-action
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Define los parametros de algoritmo Q-learning
    learning_rate = 0.1           # taza de aprendizaje
    discount_factor = 0.95        # Factor de descenso para las recompensas
    epsilon = 1                   # Probabilidad inicial de exploracion (acciones aleatorias)
    epsilon_decay_rate = 0.001    # Tasa de decaimiento de epsilon para reducir las explotacion con el tiempo
    rng = np.random.default_rng() # Generador de numeros aleatorios
    
    # Inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle principal de entrenamiento
    for i in range(episodes):

        # Ver como evoluciona el agente
        # Reinicia el entorno cada 1000 episodios, alternando entre modos con y sin renderizacion
        if(i + 1) % 1000 == 0:
            env.close()
            env = gym.make("FrozenLake-v1", render_mode = "human")
        else:
            env.close()
            env = gym.make("FrozenLake-v1")
        
        # Reinicia el entrono y obtiene el estado inicial
        state = env.reset()[0]
        
        # Variables para controlar la finalizacion del episodio
        terminated = False
        truncated = False
        
        # Bucle para cada paso dentro de un ejemplo 
        while(not terminated and not truncated):
            
            # Tomar una accion en base a si es explotacion o exploracion basado en epsilon
            if rng.random() < epsilon:
                action = env.action_space.sample()     # Exploracion:Selecciona una accion aleatoria
            else:
                action = np.argmax(q_table[state, :])  # Explotacion: Selecciona la mejor basada en Q-table
                
            # Realizar la acción y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Actualiza la tabla Q con la nueva información obtenida
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
            
            # Actualizar el estado para el siguiente paso
            state = new_state
            
        # Reduce epsilon para disminuir la exploración a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # Registra si el agente tuvo una recompensa (llegó al objetivo) en este episodio
        if reward == 1:
            rewards_por_episode[i] = 1
            
        # Imprime el progreso cada 100 episodios
        if (i + 1) % 100 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {rewards_por_episode[i]}")
    
    # Cierra el entorno al finalizar el entrenamiento
    env.close()
    
    # Imprime la tabla Q final para la inspección
    print(f"Mejor Q: {q_table}")
    
    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 100) :(t + 1)])
        
    plt.plot(suma_rewards)
    plt.show()
    
    
# Ejecutar la función de entrenamiento si el script es el programa principal
if __name__ == '__main__':
    train(20000)
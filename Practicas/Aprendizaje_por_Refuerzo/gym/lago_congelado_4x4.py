import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    env = gym.make('FrozenLake-v1')
    
    q_table = np.zeros([env.observation_space.n, env.action_space.n])\
        
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    
    # inicializa un array para almacenar las recompensas obtenidas en cada episodio
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        
        # Reinicial el entorno cada 1000 episodios, altenando entre modos con y sin renderizado
        if (i * 1) % 10000 == 0:
            env.close()
            env = gym.make('FrozenLake-v1', render_mode = 'human')
            print(f'Episodio {i + 1}')
        else:
            env.reset()
            env = gym.make('FrozenLake-v1')
               
        # Reinicia el entorno y establece el estado inicial
        state = env.reset()[0]
        done = False
        
        # Variables para controlar la finalizacion del episodio
        terminated = False
        truncated = False
        
        # bucle para cada paso dentro de un episodio
        while (not terminated and not truncated):
            # Desicion de explorar o explotar
            if rng.random() < epsilon:
                action = env.action_space.sample() # Exploracion, selecciona una accion aleatoria
            else:
                action = np.argmax(q_table[state, :]) # Explotacion, selecciona la mejor accion segun la tabla Q
                
            # Realiza la accion seleccionada y obtiene el nuevo estado y la recompensa
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Actualiza la tabla Q con la nueva informacion obtenida
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
            
            # Actualiza el estado actual para el siguiente paso
            state = new_state
            
        # Reduce el epsilon para la exploracion para la siguiente iteracion a lo largo del tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # Registra si el agente obtuvo una recompensa (llego al objetivo) en este episodio
        if reward == 1:
            rewards_per_episode[i] = 1
           
        # imprime el progreso cada 1000 episodios
        if (i + 1) % 10000 == 0:
            print(f'Episodio {i + 1} recompensa: {rewards_per_episode[i]}')
        
    # cierra el entorno al finalizar el entrenamiento
    env.close()
    
    # imprimir la mejor tabla Q obtenida
    print('Tabla Q final:')
    print(q_table)
    
    # Calcula y muestra la suma de recompensas acumuladas en bloques de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    
    plt.plot(sum_rewards)
    plt.show()
    
if __name__ == '__main__':
    train(100000)    
            
    
# Importar las librerias necesarias
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces, utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Definimos la forma del entorno
environment_rows = 11
environment_columns = 11

# Crea un arreglo 3D de numpy para alnacenar los valores Q actuales para cada par estado-acción: Q(s, a)
# EI arreglo contiene 11 filas y 11 columnas (para que coincida con la forma del entorno), así com una tercera dimensión -acción-.
# La dimensión -acción- consiste en 4 capas que nos permitirán realizar un seguimiento de los valores Q para cada acción posible en cada estado (
# EI valor de cada par (estado, acción) se Inicializa en 0.
Q_values = np.zeros((environment_rows, environment_columns, 4))

# Define las acciones
# Codigo de acción numéricos: 0 = arriba, 1 = abajo, 2 = izquierda, 3 = derecha
actions = ["up", "right", "down", "left"]

# Crea un arreglo 2D de numpy para alnacenar las recompensas de cada estado.
# EI arreglo contiene 11 filas y 11 columnas (para que coincida con la forma del entorno), y cada valor se inicializa en -100.
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100. # Establece la recompensa para el área de empaque (es decir, la meta ) en 100



# Define las ubicaciones de los pasillos (es decir, cuadros blancos) para las filas de la 1 a la 9
aisles = {}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

# Establece las recompensas para todas las ubicaciones de los pasillos (es decir, cuadros blancos)
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.
        
# Imprime la matriz de las recompensas
for row in rewards:
    print(row)
    
# Defines una function que determina si la ubicación específica es un estado terminal
def is_terminal_state(current_row_index, current_column_index):
    # Si la recompensa en la ubicación actual es -1, entonces no es un estado terminal (es decir, no es una ubicación de pasillo o de meta) es un cuadro blanco
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True
    
    
# Define una función que elegirá una ubicación de inicio aleatoria y no terminal 
def get_starting_location():
    # Obten un indice de la fila y columna aleatorio 
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    
    # Continúa eligiendo índices de fila y columna aleatorios hasta que se identifique un estado no terminal
    # (es decir, hasta que el estado elegido sea un -cuadro blanco").
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index


# Define un algoritmo epsilon greedy que elegirá qué acción tomar a continuación (es decir hacia donde moverse)
def get_next_action(current_row_index, current_column_index, epsilon):
    # Si un número aleatorio entre 0 y 1 es menor que epsilon, entonces elige una acción aleatoria
    # Si un valor elegido al azar entre 0 y 1 es menor que epsilon,
    # entonces elige el valor más prometedor de la tabla Q para este estado
    if np.random.random() < epsilon:
        return np.argmax(Q_values[current_row_index, current_column_index])
    else: # Escoge una opción aleatoria
        # Elije la acción que tenga el valor Q más alto en el estado actual
        return np.random.randint(4)
    
# Define una función que obtiene la siguiente ubicación basada en la acción elegida
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


# Define una función que obtendrá eI camino más corto entre cualquier ubicación dentro del almacén al que
# eI robot tiene pernitido viajar y la ubicación de empaque del artículo.
def get_shortes_path(start_row_index, start_column_index):
    # Retorna inmediatamente si esta es una ubicación inicial invalida
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else: # Si esta es una ubicación de inicio "legal" válida, entonces inicializa la lista de ruta
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        
        # Agrega la ubicación de inicio a la ruta
        shortest_path.append([current_row_index, current_column_index])
        
        # Continúa moviéndose a través del almacén hasta que se llegue a la ubicación de empaque
        while not is_terminal_state(current_row_index, current_column_index):
            # Obten la mejor acción a tomar
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            # Muevete a la siguiente ubicación en el camino y agrega la nueva ubicación a la lista 
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path
    

# Define los parámetros de aprendizaje para el entrenamiento
epsilon = 0.9 # Epsilon greedy, el porcetaje de veces que debemos tomar la acción (en lugar de una acción aleatoria)
discount_factor = 0.9 # Factor de descuento para las recompensas futuras
learning_rate = 0.9 # Tasa de aprendizaje para actualizar la tabla Q, la velocidad a la que el Agente de IA debe aprender

# Ejecutar 1000 episodios de entrenamiento
for episode in range(1000):
    # Obten la ubicación de inicioparaeste episodio 
    row_index, column_index = get_starting_location()
    
    # Continua tomando acciones (es decir , moviendote) hasta que llegue a un estado terminal 
    # (es decir, hasta que llegue al área de empaque del artículo o te choques con una ubicación de almacenamiento de artículos)
    while not is_terminal_state(row_index, column_index):
        # Elige que acción tomar (es decir , hacia donde moverte a continuación)
        action_index = get_next_action(row_index, column_index, epsilon)
        
        # Reelizar la acción elegida y transiciona al siguiente estado (es decir muevete a la siguiente ubicación)
        old_row_index, old_column_index = row_index, column_index # Guarda los indices de fila y columna anteriores
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        
        # Recibe la recompensa por moverte al nuevo estado y calcula la diferencia temporal 
        reward = rewards[row_index, column_index]
        old_q_value = Q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(Q_values[row_index, column_index])) - old_q_value

        # Actualiza el valor Q para el par estado-acción anterior
        new_q_value = old_q_value + (learning_rate * temporal_difference)	
        Q_values[old_row_index, old_column_index, action_index] = new_q_value
        
print("Finalmente, el valor Q para cada par estado-acción es: ")
print("Training complete!")

# Muestra algunos caminos más cortos 
print(get_shortes_path(3, 9))  # Comenzamos en al fila 3, columna 9
print(get_shortes_path(5, 0))  # Comenzamos en al fila 5, columna 0
print(get_shortes_path(9, 5))  # Comenzamos en al fila 9, columna 5

# Muestra un ejemplo de camino más corto revertido
path = get_shortes_path(5, 1) # Ir la fila 5 y clumna 2
path.reverse()
print(path)

# Obtenen la matriz de recompensas
rewards_matrix = rewards

# Ejecuta el algoritmo de entrenamiento y obten los valores de Q 

Q_values_matrix = np.max(Q_values, axis=2)

# Muestra la natriz de recompensas como una imagen utilizando matplotlib
# Muestra la matriz de valores Q como una imagen  utilizando matplotlib
plt.imshow(rewards_matrix, cmap='Blues', interpolation='nearest')
plt.title("Rewards Matriz")
plt.show()
plt.imshow(Q_values_matrix, cmap='Greens', interpolation='nearest')
plt.title("Q Values Matriz - Results Matrix")
plt.show()

# Obten el camino más corto desde el punto de inicio 
start_row = 9
start_column = 0
shortest_path = get_shortes_path(start_row, start_column)

# Crea una matriz vacía para visualizar el recorrido completo 
shortest_path_full_matrix = np.zeros_like(rewards_matrix)


# Marca el camino completo en la matriz 
for i in range(len(shortest_path) - 1):
    current_step = shortest_path[i]
    next_step = shortest_path[i + 1]
    current_row, current_column = current_step
    next_row, next_column = next_step
    shortest_path_full_matrix[current_row, current_column] = 1
    

# Marca la ubicación final (punto de empaque) en la amtriz
shortest_path_full_matrix[shortest_path[-1][0], shortest_path[-1][1]] = 1

# Función para actualizar el gráfico en cada cuadro  de animación
def update_frame(frame):
    plt.cla()  # limpia el gráfico actual
    plt.imshow(shortest_path_full_matrix, cmap='copper', interpolation='nearest')
    plt.title("Full Path from Start")
    current_step = shortest_path[frame]
    plt.scatter(current_step[1], current_step[0], marker='o', color='green')
    
    
# Configuración de la animación
fig = plt.figure()
ani = animation.FuncAnimation(fig, update_frame, frames=len(shortest_path), interval=500, repeat=False)

# Mostrar la aniamción
plt.show()

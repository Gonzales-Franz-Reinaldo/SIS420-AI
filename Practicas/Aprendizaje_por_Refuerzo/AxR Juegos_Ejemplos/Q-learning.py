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
aisles[5] = [i for i in range(1, 11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(1, 11)]

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

# Minuto 5:20
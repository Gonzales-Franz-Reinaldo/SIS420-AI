"""
Descripción del script: Este script implementa la lógica del juego para un agente de aprendizaje por refuerzo (RL) en Tetris.
Autor: Daniel Rodriguez
CI: V-19.333.348
Fecha: 01/12/2023
"""

# Importaciones
import os
import pickle
import random
import numpy as np
import tensorflow as tf

# Importación de constantes desde un módulo separado
from src.constants.constants import ACTION_SIZE, EPSILON, EPSILON_DECAY, EPSILON_MIN, GAMMA, LEARNING_RATE, MEMORY_PATH, MODEL_PATH, STATE_SIZE

# Configuración de visibilidad de dispositivos CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class RLAgent:
    def __init__(self):
        # Parámetros del agente
        self.stateSize = STATE_SIZE  # Tamaño del vector de estado
        self.actionSize = ACTION_SIZE  # Número de acciones posibles (izquierda, derecha, rotar, bajar)
        self.memory = []  # Almacenamiento de la memoria de experiencia
        self.gamma = GAMMA  # Factor de descuento para las recompensas futuras
        self.epsilon = EPSILON  # Factor de exploración inicial
        self.epsilonDecay = EPSILON_DECAY  # Tasa de decaimiento del factor de exploración
        self.epsilonMin = EPSILON_MIN  # Valor mínimo del factor de exploración
        self.learningRate = LEARNING_RATE  # Tasa de aprendizaje

        # Carga la memoria
        self.loadMemory()

        # Crea y compila la red neuronal
        self.model = self.buildModel()
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learningRate))

    def saveMemory(self, filepath = MEMORY_PATH):
        """
        Guarda la memoria en un archivo pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.memory, f)

    def loadMemory(self, filepath=MEMORY_PATH):
        """
        Carga la memoria desde un archivo pickle.
        """
        try:
            with open(filepath, 'rb') as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            # Manejar el caso en que el archivo no existe
            print(f"El archivo {filepath} no existe. La memoria no se cargó.")
        except Exception as e:
            # Manejar otras excepciones que puedan ocurrir durante la carga
            print(f"Error al cargar la memoria desde {filepath}: {e}")


    def buildModel(self):
        """
        Construye el modelo de red neuronal.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.stateSize, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.actionSize, activation='linear'))
        return model
    
    def generateAction(self, state):
        """
        Genera una acción basada en la exploración epsilon-greedy.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actionSize)
        
        return np.argmax(self.model.predict(state)[0])
    
    def replay(self, batchSize):
        """
        Realiza el proceso de replay para actualizar la red neuronal.
        """
        if len(self.memory) < batchSize:
            return
        
        minibatch = random.sample(self.memory, batchSize)
        targetsList = []
        statesReshapedList = []
        
        for state, action, reward, nextState, done in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(nextState.reshape((1, -1)))[0]) if not done else reward
            targetF = self.model.predict(state.reshape((1, -1)))
            targetF[0][action] = target
            targetsList.append(targetF)
            statesReshapedList.append(state.reshape((1, -1)))
        
        self.model.fit(np.vstack(statesReshapedList), np.vstack(targetsList), epochs=1, verbose=0)

    def saveModel(self, filepath = MODEL_PATH):
        """
        Guarda el modelo en un archivo HDF5.
        """
        self.model.save(filepath)

    def loadModel(self, filepath = MODEL_PATH):
        """
        Carga el modelo desde un archivo HDF5.
        """
        # Verifica si el archivo existe
        if not os.path.exists(filepath):
            print(f"El archivo {filepath} no existe. No se cargó el modelo.")
            return
        # Establece los pesos en la red neuronal
        self.model = tf.keras.models.load_model(filepath)
    
    def remember(self, state, action, reward, nextState, done):
        """
        Almacena la experiencia en la memoria temporal y guarda la memoria en un archivo.
        """
        # Agrega la transición a la memoria temporal
        self.memory.append((state, action, reward, nextState, done))
        # Guarda la memoria en un archivo
        self.saveMemory()
        
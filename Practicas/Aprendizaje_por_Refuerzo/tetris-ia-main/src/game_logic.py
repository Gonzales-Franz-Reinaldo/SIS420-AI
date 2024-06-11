"""
Descripción del script: Este script implementa la lógica del juego para un agente de aprendizaje por refuerzo (RL) en Tetris.
Autor: Daniel Rodriguez
CI: V-19.333.348
Fecha: 01/12/2023
"""

# Importaciones
import random
import numpy as np
import pygame

# Importación de constantes y clases desde módulos separados
from src.constants.constants import BLOCK_SIZE, COLORS, COLS, HEIGHT, ROWS, SHAPES, TEXT_WINDOW_TITLE, WIDTH
from src.classes.direction import Direction
from src.classes.piece import Piece

class RLGameLogic:
    def __init__(self):
        # Inicialización de Pygame y configuración de la ventana del juego
        pygame.init()
        self.playHeight = ROWS * BLOCK_SIZE
        self.playWidth = COLS * BLOCK_SIZE
        self.block_size = BLOCK_SIZE
        self.screenHeight = HEIGHT
        self.screenWidth = WIDTH

        self.topLeftX = (self.screenWidth - self.playWidth) // 2
        self.topLeftY = self.screenHeight - self.playHeight

        self.shapes = SHAPES
        self.colors = COLORS

        # Configuración de la ventana de Pygame
        self.win = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption(TEXT_WINDOW_TITLE)

    def generateGrid(self, lockedPositions={}):
        """
        Genera la cuadrícula del juego a partir de las posiciones bloqueadas.
        """
        grid = [[(0, 0, 0) for _ in range(COLS)] for _ in range(ROWS)]

        for (x, y) in lockedPositions:
            color = lockedPositions[(x, y)]
            grid[y][x] = color

        return grid

    def convertShapeFormat(self, shape):
        """
        Convierte el formato de la forma para obtener sus posiciones en la cuadrícula.
        """
        positions = []
        format = shape.value[shape.rotation % len(shape.value)]

        for i, row in enumerate(format):
            for j, column in enumerate(list(row)):
                if column == '0':
                    positions.append((shape.x + j, shape.y + i))

        for i, pos in enumerate(positions):
            positions[i] = (pos[0] - 2, pos[1] - 4)

        return positions

    def validSpace(self, shape, grid):
        """
        Verifica si hay espacio válido para la forma en la cuadrícula.
        """
        acceptedPositions = [[(j, i) for j in range(COLS) if grid[i][j] == (0, 0, 0)] for i in range(ROWS)]
        acceptedPositions = [j for sub in acceptedPositions for j in sub]
        formatted = self.convertShapeFormat(shape)

        for pos in formatted:
            if pos not in acceptedPositions:
                if pos[1] > -1:
                    return False

        return True

    def checkLost(self, lockedPositions):
        """
        Verifica si el jugador ha perdido.
        """
        for pos in lockedPositions:
            if pos[1] < 1:
                return True
        return False

    def getShape(self):
        """
        Retorna una nueva pieza aleatoria.
        """
        return Piece(5, 0, random.choice(self.shapes))

    def clearRows(self, grid, locked):
        """
        Limpia las filas completas en la cuadrícula.
        """
        rowsCleared = 0
        for i in range(len(grid)-1, -1, -1):
            row = grid[i]
            if (0, 0, 0) not in row:
                rowsCleared += 1
                ind = i
                for j in range(len(row)):
                    try:
                        del locked[(j, i)]
                    except:
                        continue

        if rowsCleared > 0:
            for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
                x, y = key
                if y < ind:
                    new_key = (x, y + rowsCleared)
                    locked[new_key] = locked.pop(key)

        return rowsCleared
    
    def reset(self):
        """
        Reinicia el estado del juego.
        """
        self.currentPiece = self.getShape()
        self.lockedPositions = {}
        self.grid = self.generateGrid(self.lockedPositions)

    def getState(self, currentPiece = 0):
        """
        Obtiene el estado actual del juego para el agente.
        """
        if not currentPiece == 0:
            yPosition = currentPiece.y
        else:
            yPosition = currentPiece

        stateVector = np.array([yPosition / self.screenHeight])

        if len(stateVector) < 10:
            stateVector = np.pad(stateVector, (0, 10 - len(stateVector)))

        stateVector = np.reshape(stateVector, (1, 10))

        return stateVector
    
    def handleIAEvents(self, currentPiece, grid, action):
        """
        Maneja los eventos de la IA para mover la pieza.
        """
        if action == 0:
            currentPiece.x -= 1
            if not self.validSpace(currentPiece, grid):
                currentPiece.x += 1

        elif action == 1:
            currentPiece.x += 1
            if not self.validSpace(currentPiece, grid):
                currentPiece.x -= 1

        elif action == 2:
            currentPiece.rotation = (currentPiece.rotation + 1) % len(currentPiece.value)
            if not self.validSpace(currentPiece, grid):
                currentPiece.rotation = (currentPiece.rotation - 1) % len(currentPiece.value)

        elif action == 3:
            currentPiece.y += 1
            if not self.validSpace(currentPiece, grid):
                currentPiece.y -= 1
    
    def handlePlayerEvents(self, currentPiece, grid):
        """
        Maneja los eventos del jugador para mover la pieza.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == Direction.LEFT.value:
                    currentPiece.x -= 1
                    if not self.validSpace(currentPiece, grid):
                        currentPiece.x += 1

                elif event.key == Direction.RIGHT.value:
                    currentPiece.x += 1
                    if not self.validSpace(currentPiece, grid):
                        currentPiece.x -= 1

                elif event.key == Direction.UP.value:
                    # rotate shape
                    currentPiece.rotation = (currentPiece.rotation + 1) % len(currentPiece.value)
                    if not self.validSpace(currentPiece, grid):
                        currentPiece.rotation = (currentPiece.rotation - 1) % len(currentPiece.value)

                if event.key == Direction.DOWN.value:
                    # move shape down
                    currentPiece.y += 1
                    if not self.validSpace(currentPiece, grid):
                        currentPiece.y -= 1
                    
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

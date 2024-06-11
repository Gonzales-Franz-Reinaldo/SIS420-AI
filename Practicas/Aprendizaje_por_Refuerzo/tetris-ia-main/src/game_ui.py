"""
Descripción del script: Este script implementa la lógica del juego para un agente de aprendizaje por refuerzo (RL) en Tetris.
Autor: Daniel Rodriguez
CI: V-19.333.348
Fecha: 01/12/2023
"""

# Importaciones
import pygame
import numpy as np

# Importación de clases y constantes desde módulos separados
from src.classes.color import Color
from src.constants.constants import BLOCK_SIZE, COLS, ROWS, TEXT_END_GAME, TEXT_FONT, TEXT_GAME_TITLE, TEXT_NEXT, TEXT_SCORE, TEXT_START

# Inicialización de Pygame para el uso de fuentes de texto
pygame.font.init()

class GameUI:
    def __init__(self, gameLogic, agent):
        # Inicialización de la interfaz con la lógica del juego y el agente de RL
        self.gameLogic = gameLogic
        self.agent = agent

    def printOnScreen(self, text, size, color):
        """
        Imprime texto en la pantalla.
        """
        font = pygame.font.SysFont(TEXT_FONT, size, bold=True)
        label = font.render(text, 1, color)

        self.gameLogic.win.blit(label, (self.gameLogic.topLeftX + self.gameLogic.playWidth / 2 - (label.get_width() / 2),
                                        self.gameLogic.topLeftY + self.gameLogic.playHeight / 2 - label.get_height() / 2))

    def drawGrid(self):
        """
        Dibuja la cuadrícula del juego en la pantalla.
        """
        showX = self.gameLogic.topLeftX
        showY = self.gameLogic.topLeftY

        for i in range(ROWS):
            pygame.draw.line(self.gameLogic.win, Color.GRAY.value, (showX, showY + i * BLOCK_SIZE), (showX + self.gameLogic.playWidth, showY + i * BLOCK_SIZE))
        for j in range(COLS):
            pygame.draw.line(self.gameLogic.win, Color.GRAY.value, (showX + j * BLOCK_SIZE, showY), (showX + j * BLOCK_SIZE, showY + self.gameLogic.playHeight))

    def drawNextShape(self, shape):
        """
        Dibuja la próxima pieza que aparecerá en la pantalla.
        """
        font = pygame.font.SysFont(TEXT_FONT, 30)
        label = font.render(TEXT_NEXT, 1, Color.WHITE.value)

        showX = self.gameLogic.topLeftX + self.gameLogic.playWidth + 50
        showY = self.gameLogic.topLeftY + self.gameLogic.playHeight / 2 - 150
        format = shape.value[shape.rotation % len(shape.value)]

        for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    pygame.draw.rect(self.gameLogic.win, shape.color, (showX + 20 + j * BLOCK_SIZE, showY + 50 + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

        self.gameLogic.win.blit(label, (showX + COLS, showY))

    def drawScore(self, score):
        """
        Dibuja el puntaje en la pantalla.
        """
        font = pygame.font.SysFont(TEXT_FONT, 30)
        label = font.render(TEXT_SCORE, 1, Color.WHITE.value)

        showInX = self.gameLogic.topLeftX - self.gameLogic.playWidth + 100
        showInY1 = self.gameLogic.topLeftY + self.gameLogic.playHeight / 2 - 150
        showInY2 = self.gameLogic.topLeftY + self.gameLogic.playHeight / 2 - 100
        self.gameLogic.win.blit(label, (showInX, showInY1))

        scoreValue = font.render(str(score), 1, Color.GREEN.value)
        self.gameLogic.win.blit(scoreValue, (showInX + 60, showInY2))
    
    def drawAgentRewards(self, reward):
        """
        Dibuja las recompensas del agente en la pantalla.
        """
        font = pygame.font.SysFont(TEXT_FONT, 30)
        label = font.render('REWARD', 1, Color.WHITE.value)

        showInX = self.gameLogic.topLeftX - self.gameLogic.playWidth + 100
        showInY1 = self.gameLogic.topLeftY + self.gameLogic.playHeight / 2 - 50
        showInY2 = self.gameLogic.topLeftY + self.gameLogic.playHeight / 2
        self.gameLogic.win.blit(label, (showInX, showInY1))

        scoreValue = font.render(str(reward), 1, Color.RED.value)
        self.gameLogic.win.blit(scoreValue, (showInX + 60, showInY2))

    def drawWindow(self, grid):
        """
        Dibuja la ventana del juego en la pantalla.
        """
        self.gameLogic.win.fill((0, 0, 0))
        font = pygame.font.SysFont(TEXT_FONT, 40)
        label = font.render(TEXT_GAME_TITLE, 1, Color.WHITE.value)

        self.gameLogic.win.blit(label, (self.gameLogic.topLeftX + self.gameLogic.playWidth / 2 - (label.get_width() / 2), 30))

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                pygame.draw.rect(self.gameLogic.win, grid[i][j], (self.gameLogic.topLeftX + j * BLOCK_SIZE, self.gameLogic.topLeftY + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

        self.drawGrid()
        pygame.draw.rect(self.gameLogic.win, Color.GRAY.value, (self.gameLogic.topLeftX, self.gameLogic.topLeftY, self.gameLogic.playWidth, self.gameLogic.playHeight), 5)

    def runGame(self):
        """
        Ejecuta el juego.
        """
        # Inicialización de variables y estructuras de datos necesarias
        lockedPositions = {} # (x,y):(r,g,b)
        grid = self.gameLogic.generateGrid(lockedPositions)
        currentPiece = self.gameLogic.getShape()
        nextPiece = self.gameLogic.getShape()
        clock = pygame.time.Clock()

        fallSpeed = 0.27
        levelTime = 0
        fallTime = 0
        reward = 0
        score = 0

        changePiece = False
        run = True

        while run:
            # Actualización de la cuadrícula del juego y el tiempo de juego
            grid = self.gameLogic.generateGrid(lockedPositions)
            fallTime += clock.get_rawtime()
            levelTime += clock.get_rawtime()
            clock.tick()

            # Ajuste de la velocidad de caída según el tiempo de juego
            if levelTime / 1000 > 4:
                levelTime = 0
                if fallSpeed > 0.15:
                    fallSpeed -= 0.005

            # Control de la caída de la pieza actual
            if fallTime / 1000 >= fallSpeed:
                fallTime = 0
                currentPiece.y += 1
                # Verificación de la validez de la posición de la pieza actual
                if not (self.gameLogic.validSpace(currentPiece, grid)) and currentPiece.y > 0:
                    currentPiece.y -= 1
                    changePiece = True

            # Manejo de eventos del jugador
            self.gameLogic.handlePlayerEvents(currentPiece, grid)
            
            # IA IMPLEMENTATION

            # Obtención del estado actual para la IA
            currentState = self.gameLogic.getState(currentPiece)

            # Selección de acción por la IA (exploración o explotación)
            if np.random.rand() <= self.agent.epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = self.agent.generateAction(currentState)

            # Aplicación de la acción seleccionada por la IA
            self.gameLogic.handleIAEvents(currentPiece, grid, action)

            # END IA IMPLEMENTATION

            # Conversión de la posición de la forma actual y actualización de la cuadrícula
            shapePosition = self.gameLogic.convertShapeFormat(currentPiece)

            for i in range(len(shapePosition)):
                x, y = shapePosition[i]
                if y > -1:
                    grid[y][x] = currentPiece.color

            # Cambio de pieza si ha llegado al final
            if changePiece:
                for pos in shapePosition:
                    p = (pos[0], pos[1])
                    lockedPositions[p] = currentPiece.color
                currentPiece = nextPiece
                nextPiece = self.gameLogic.getShape()
                changePiece = False
                
                # Limpieza de filas completas y actualización del puntaje
                rowsCleared = self.gameLogic.clearRows(grid, lockedPositions)
                if rowsCleared > 0:
                    score += (10 * rowsCleared)
                    reward += 1

            # Obtención del estado siguiente y registro de la transición en la memoria
            nextState = self.gameLogic.getState(currentPiece)
            nextState = np.array(nextState)
            self.agent.remember(currentState, action, reward, nextState, not self.gameLogic.validSpace(currentPiece, grid))
            
            # Dibujo de la ventana del juego y actualización de la interfaz
            self.drawWindow(grid)
            self.drawNextShape(nextPiece)
            self.drawScore(score)
            self.drawAgentRewards(reward)
            pygame.display.update()

            # Entrenamiento del modelo de la IA y actualización del factor de exploración
            self.agent.replay(batchSize = 8)
            self.agent.epsilon = max(self.agent.epsilonMin, self.agent.epsilon * self.agent.epsilonDecay)

            # Verificación de fin de juego
            if self.gameLogic.checkLost(lockedPositions):
                run = False

        # Pantalla de fin de juego y espera antes de cerrar la ventana
        self.gameLogic.win.fill((0, 0, 0))
        self.printOnScreen(TEXT_END_GAME, 40, Color.WHITE.value)
        pygame.display.update()
        pygame.time.delay(3000)

    def start(self):
        """
        Inicia el juego y la interfaz de usuario.
        """
        run = True

        # Carga el modelo al iniciar el juego si existe
        # self.agent.loadModel()

        self.gameLogic.win.fill((0, 0, 0))
        self.printOnScreen(TEXT_START, 20, Color.WHITE.value)
        pygame.display.update()
        pygame.time.delay(3000)

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Inicia el juego
            self.runGame()

            # Guarda el modelo al finalizar el juego
            # self.agent.saveModel()

        pygame.quit()
        quit()
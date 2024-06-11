"""
Descripción del script: Este script implementa la lógica del juego para un agente de aprendizaje por refuerzo (RL) en Tetris.
Autor: Daniel Rodriguez
CI: V-19.333.348
Fecha: 01/12/2023
"""
# Importación de clases
from src.game_logic import RLGameLogic
from src.game_agent import RLAgent
from src.game_ui import GameUI

def main():
    agent = RLAgent()
    gameLogic = RLGameLogic()
    gameUI = GameUI(gameLogic, agent)
    gameUI.start()

if __name__ == "__main__":
    main()
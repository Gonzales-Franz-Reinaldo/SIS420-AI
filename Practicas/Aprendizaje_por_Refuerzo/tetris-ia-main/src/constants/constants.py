from src.classes.tetromino import Tetromino
from src.classes.color import Color

SHAPES = [
    Tetromino.S.value,
    Tetromino.Z.value,
    Tetromino.I.value,
    Tetromino.O.value,
    Tetromino.J.value,
    Tetromino.L.value,
    Tetromino.T.value
]

COLORS = [
    Color.GREEN.value,
    Color.RED.value,
    Color.CYAN.value,
    Color.YELLOW.value,
    Color.ORANGE.value,
    Color.BLUE.value,
    Color.VIOLET.value,
]

TEXT_START = 'TETRIS - Daniel Rodríguez. CI 19.333.348'
TEXT_END_GAME = 'Juego terminado'
TEXT_WINDOW_TITLE = 'Tetris IA'
TEXT_GAME_TITLE = 'TETRIS IA'
TEXT_NEXT = 'SIGUIENTE'
TEXT_FONT = 'comicsans'
TEXT_SCORE = 'PUNTAJE'
BLOCK_SIZE = 30
HEIGHT = 700
WIDTH = 800
ROWS = 20
COLS = 10

# AGENT

MEMORY_PATH = 'src/database/memory.pkl'
MODEL_PATH = 'src/database/model.h5'

STATE_SIZE = 10
ACTION_SIZE = 4
GAMMA = 0.95
EPSILON = 0.5
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LEARNING_RATE = 0.1
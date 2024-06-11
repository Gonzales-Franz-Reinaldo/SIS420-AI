from src.constants.constants import COLORS, SHAPES

class Piece:
    def __init__(self, column, row, shape):
        self.color = COLORS[SHAPES.index(shape)]
        self.value = shape
        self.rotation = 0
        self.x = column
        self.y = row
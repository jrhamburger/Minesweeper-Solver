import numpy as np
import random

class Minesweeper:
    def __init__(self, rows=9, cols = 9, mine_count = 10):
        self.rows = rows
        self.cols = cols
        self.mine_count = mine_count

        self.board = np.zeros((rows, cols), dtype=int)
        self.revealed = np.zeros((rows, cols), dtype=bool)
        self.flagged = np.zeros((rows, cols), dtype=bool)

    def get_neighbors(self, row, col):
        neighbors = []

        for row_offset in [-1, 0, 1]:
            for col_offset in [-1, 0, 1]:
                if row_offset == 0 and col_offset == 0:
                     continue

                neighbor_row = row + row_offset
                neighbor_col = col + col_offset

                if (0 <= neighbor_row < self.rows
                    and 0 <= neighbor_col < self.cols):
                    neighbors.append((neighbor_row, neighbor_col))

        return neighbors

    def place_mines(self, first_row, first_col):
        protected = self.get_neighbors(first_row, first_col)
        protected.append((first_row, first_col))

        placed = 0

        while placed < self.mine_count:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)

            if (row, col) in protected:
                continue

            if self.board[row][col] == -1:
                continue

            self.board[row][col] = -1
            placed += 1

    def calculate_numbers(self):

        for row in range(self.rows):
            for col in range(self.cols):

                neighbors = self.get_neighbors(row, col)





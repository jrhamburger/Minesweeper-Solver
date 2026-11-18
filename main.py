from game import Minesweeper

game = Minesweeper()

print(game.board)

print(game.get_neighbors(0, 0))
print(game.get_neighbors(0, 4))
print(game.get_neighbors(4, 4))
import cv2
from Processing_puzzle import Puzzle as p
from Processing_puzzle import sides_finder as sf

puzzle = p.Puzzle()
puzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = puzzle.get_pieces()


for idx, piece in enumerate(pieces):
    img = piece.get_black_white_image()

    side1, side2, side3, side4 = piece.get_4_sides()
    print(side1)










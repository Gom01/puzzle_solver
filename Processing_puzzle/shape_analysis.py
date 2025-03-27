from Processing_puzzle import Puzzle as p
import matplotlib.pyplot as plt
import numpy as np

def find_male_female(puzzle):
    pieces = puzzle.get_pieces()

    for idx, piece in enumerate(pieces):
        contour = piece.get_contours()
        cx, cy = piece.x, piece.y


        #side1, side2, side3, side4 = piece.get_side()

        values_x, values_y = [], []


        #parcourir le slide
        #filtrer les points (enlever la moitié)
        #si j'ai un grand changement au niveau du x ou y je garde



        print(values_x)
        print(values_y)
        print()
        plt.plot(values_x, values_y , 'ro')
        plt.plot(cx, cy , 'ro')
        plt.show()


    return()


myPuzzle = p.Puzzle()
myPuzzle.load_puzzle("../Processing_puzzle/res/puzzle.pickle")
find_male_female(myPuzzle)

from turtledemo.paint import switchupdown
from unittest import case

import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev

from Processing_puzzle import Puzzle as p

def piece_definition(myPuzzle):

    def piece_hierarchy(piece, windows = False):

        sides = piece.get_sides()


        counter_droit = 0

        for side in sides:
            if(side.get_side_info() == "droit"):
                counter_droit += 1

        match counter_droit:
            case 0: side.set_side_hierarchy("inside")
            case 1: side.set_side_hierarchy("border")
            case 2: side.set_side_hierarchy("angle")

        if windows:
            print(sides)

        return



    pieces = myPuzzle.get_pieces()

    for i, piece in enumerate(pieces):
        sides_eq = piece_hierarchy(piece, False)

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("All additional info for sides saved!")
    return ()
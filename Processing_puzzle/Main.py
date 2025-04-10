import Puzzle as puzzle
import cv2

from Processing_puzzle.parsing.straight_piece import straighten_piece
from Processing_puzzle.parsing.color_analysis import find_color
from Processing_puzzle.parsing.corners import find_corners
from Processing_puzzle.parsing.parse import parse_image
from Processing_puzzle.parsing.sides_analysis import sides_information
from Processing_puzzle.parsing.sides_finder import find_sides


image_path = '../images/p1_b/Natel.Black.jpg'


##!! When taking the picture no piece should be sticked together (let some space) and use black tissue
myPuzzle = puzzle.Puzzle()

parse_image(image_path, myPuzzle)
find_corners(myPuzzle)
find_sides(myPuzzle)
find_color(myPuzzle)
sides_information(myPuzzle)
myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')

pieces = myPuzzle.get_pieces()
#print()
#       img = piece.get_color_image()
#       #straighten_piece(piece)
#       cv2.imshow('rotated image', img)
#       cv2.waitKey(0)

#     img_black = piece.image_black_white
#     contours = piece.contours
#     i = piece.index
#     corners = piece.get_corners()
#     contours_col = piece.get_color_contour()
#     sides = piece.get_sides()

    # for idx, side in enumerate(sides):
    #     colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    #     for pt in side:
    #         cv2.circle(img, pt, 7, colors[idx], -1)

    # #Show colors
    # for idx , pt in enumerate(contours):
    #     b,g,r = contours_col[idx]
    #     cv2.circle(img, pt, 7, (int(b), int(g), int(r)), -1)

    # cv2.imshow('Corner detection', img)
    # cv2.waitKey(0)





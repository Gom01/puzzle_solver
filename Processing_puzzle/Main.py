from Processing_puzzle import puzzle_parsing as p
import Puzzle as puzzle
from Processing_puzzle import corner_finder as cf
from Processing_puzzle import color_analysis as ca
from Processing_puzzle import sides_finder as sf
import cv2
image_path = '../images/p1_b/Natel.Black.jpg'


##!! When taking the picture no piece should be sticked together (let some space) and use black tissue
myPuzzle = puzzle.Puzzle()

p.parse_image(image_path, myPuzzle)
cf.find_corners(myPuzzle)
#ca.find_color(myPuzzle)
#sf.find_sides(myPuzzle)

pieces = myPuzzle.get_pieces()
print()
for idx, piece in enumerate(pieces):
    img = piece.image_color
    img_black = piece.image_black_white
    contours = piece.contours
    i = piece.index

    #print(f"Piece number {i}: {contours}")
    #cv2.imshow('Corner detection', img_black)
    #cv2.waitKey(0)
    #cv2.imshow('Corner detection', img)
    #cv2.waitKey(0)





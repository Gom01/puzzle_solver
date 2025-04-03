from Processing_puzzle import puzzle_parsing as p
from Processing_puzzle import corner_finder as cf
from Processing_puzzle import color_analysis as ca
from Processing_puzzle import sides_finder as sf
from Processing_puzzle import positive_negative_sides_finder as sp

image_path = '../images/p1_b/Natel.Black.jpg'

myPuzzle = p.Puzzle()

p.parse_image(image_path, myPuzzle)
cf.find_corners(myPuzzle)
ca.find_color(myPuzzle)
sf.find_sides(myPuzzle)
sp.sides_information(myPuzzle)


pieces = myPuzzle.get_pieces()
print()
for idx, piece in enumerate(pieces):
    print(piece)





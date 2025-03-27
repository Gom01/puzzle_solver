from Processing_puzzle import puzzle_parsing as p
from Processing_puzzle import corner_finder as cf

image_path = '../images/p1_b/Natel.Black.jpg'

myPuzzle = p.Puzzle()

p.parse_image(image_path, myPuzzle)
cf.find_corners(myPuzzle)

pieces = myPuzzle.get_pieces()
for idx, piece in enumerate(pieces):
    print(piece)





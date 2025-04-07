import cv2 as cv

from Processing_puzzle import Puzzle as p

# Charger et traiter le puzzle
myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()



piece = pieces[0]
colors = piece.get_sides()[0].get_side_colors()
points = piece.get_sides()[0].get_side_points()

img = piece.get_black_white_image().copy()

# Iterate through the points and corresponding colors
for i, point in enumerate(points):
    # Ensure the color is a tuple of integers (B, G, R) with values in range [0, 255]
    color = tuple(map(int, colors[i]))  # Convert color to tuple of integers

    # Draw a circle at each point with the corresponding color
    cv.circle(img, point, 5, color, -1)  # Radius = 5, thickness = -1 for filled circle

# Show the image with the drawn circles
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()




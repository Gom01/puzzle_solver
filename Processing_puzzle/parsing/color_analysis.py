from Processing_puzzle import Puzzle as p
import numpy as np
import cv2

'''find_color() samples the color along each side of the puzzle piece.
    Each side will have individual colors per point, optionally offset inward by a factor.'''
def find_color(puzzle, factor=0):
    pieces = puzzle.get_pieces()
    for idx, piece in enumerate(pieces):
        sides = piece.get_sides()
        color_image = piece.get_color_image().copy()
        height, width = color_image.shape[:2]
        color_sides = []

        # Use full contour to compute centroid
        contour = piece.get_contours()
        M = cv2.moments(np.array(contour))
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            print(f"Center not found for piece {idx}")
            cx, cy = width // 2, height // 2

        piece.x, piece.y = cx, cy  # Store centroid in the piece

        # For each side of the piece
        for side in sides:
            contour_side = side.get_side_contour()
            side_colors = []
            for point in contour_side:
                x, y = point

                # Offset the point slightly toward centroid
                move_x = int((cx - x) * factor)
                move_y = int((cy - y) * factor)
                new_x = np.clip(x + move_x, 0, width - 1)
                new_y = np.clip(y + move_y, 0, height - 1)

                b, g, r = color_image[new_y, new_x]
                side_colors.append([int(b), int(g), int(r)])
                cv2.circle(color_image, (new_x, new_y), 1, (int(b), int(g), int(r)), -1)

            side.set_side_color(side_colors)

        # cv2.imshow('img', color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Save the colors of the four sides

    puzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Colored contour saved!")

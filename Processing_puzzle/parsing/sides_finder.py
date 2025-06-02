import cv2 as cv
import numpy as np

from Processing_puzzle.Sides import Side

def find_sides(myPuzzle, window=False):
    pieces = myPuzzle.get_pieces()

    for i, piece in enumerate(pieces):
        corners = piece.get_corners()

        if corners[0] != [2, 2]:
            print(f"🔄 Processing piece {i + 1}/{len(pieces)}", end='\r')

            side1, side2, side3, side4 = [], [], [], []
            contour = piece.get_contours()

            def find_nearest_index(contour, target_point):
                return min(range(len(contour)),
                           key=lambda i: np.linalg.norm(np.array(contour[i]) - np.array(target_point)))

            start_index = find_nearest_index(contour, corners[0])

            current_side = side1
            visited_corners = {tuple(corners[0])}
            point_origin = contour[start_index]
            index = start_index

            while True:
                point = contour[index]
                point_tuple = tuple(point)

                for corner in corners:
                    if np.linalg.norm(np.array(point) - np.array(corner)) < 2 and tuple(corner) not in visited_corners:
                        visited_corners.add(tuple(corner))

                        if len(visited_corners) == 2:
                            current_side = side2
                        elif len(visited_corners) == 3:
                            current_side = side3
                        elif len(visited_corners) == 4:
                            current_side = side4
                        break

                current_side.append(point)
                index = (index + 1) % len(contour)

                if len(visited_corners) == 4 and np.linalg.norm(np.array(point) - np.array(corners[0])) < 2:
                    break

            if window:
                img = piece.get_color_image().copy()
                corner1, corner2, corner3, corner4 = corners
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

                for i, side in enumerate([side1, side2, side3, side4]):
                    for j in range(len(side) - 1):
                        cv.line(img, side[j], side[j + 1], colors[i], 4)


                for corner in [corner1, corner2, corner3, corner4]:
                    cv.circle(img, corner, radius=5, color=(255, 90, 0), thickness=-1)

                cv.imshow("4 Sides", img)
                cv.waitKey(0)
                cv.destroyAllWindows()

            # Set sides in order: top, right, bottom, left
            piece.set_sides(
                Side(side4, piece.get_color_image()),
                Side(side3, piece.get_color_image()),
                Side(side2, piece.get_color_image()),
                Side(side1, piece.get_color_image())
            )
        else:
            piece.set_sides(
                Side([[2,2]], piece.get_color_image()),
                Side([[2,2]], piece.get_color_image()),
                Side([[2,2]], piece.get_color_image()),
                Side([[2,2]], piece.get_color_image())
            )
            print(f"❌ No corners found for piece {i}, skipping.")

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')

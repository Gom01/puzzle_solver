import cv2
import numpy as np
from Processing_puzzle import Puzzle as p

def sides_information(myPuzzle, windows=False):

    def type_of_sides(piece, windows):
        sides = piece.get_sides()
        if sides[0].get_side_contour() == [[2,2]]:
            print(f"Skipping piece {piece.get_index()}")
            for side in sides:
                side.set_side_info(2)
            return

        img = piece.get_color_image().copy()

        piece_contour = np.array(piece.get_contours()).reshape((-1, 1, 2))
        M1 = cv2.moments(piece_contour)
        piece.set_moment(M1)

        x_M1 = int(M1["m10"] / M1["m00"]) if M1["m00"] != 0 else 0
        y_M1 = int(M1["m01"] / M1["m00"]) if M1["m00"] != 0 else 0

        for i, side in enumerate(sides):
            side_array = np.array(side.get_side_contour()).reshape((-1, 1, 2))
            hull = cv2.convexHull(side_array)

            M = cv2.moments(hull)
            x_M = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            y_M = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            x1, y1 = side_array[0][0]
            x2, y2 = side_array[-1][0]

            if x2 - x1 != 0:
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1
                position_relative = lambda x, y: y - (a * x + b)
            else:
                position_relative = lambda x, y: x - x1

            pos_M1 = position_relative(x_M1, y_M1)
            pos_M = position_relative(x_M, y_M)

            piece_area = cv2.contourArea(piece_contour)
            hull_area = cv2.contourArea(hull)
            area_threshold = piece_area * 0.10

            if hull_area < area_threshold:
                type_cote = "droit"
                side.set_side_info(0)
                color = (255, 255, 0)
            elif np.sign(pos_M1) == np.sign(pos_M):
                type_cote = "concave"
                side.set_side_info(-1)
                color = (255, 0, 0)
            else:
                type_cote = "convexe"
                side.set_side_info(1)
                color = (0, 255, 0)

            if windows:
                print(f"Le côté {i} est {type_cote}.")
                cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)
                cv2.circle(img, (x_M1, y_M1), 5, (255, 0, 0), -1)
                cv2.circle(img, (x_M, y_M), 5, (0, 0, 255), -1)
                cv2.circle(img, (x1, y1), 5, (255, 255, 255), -1)
                cv2.circle(img, (x2, y2), 5, (255, 255, 255), -1)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if windows:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for side in sides:
                contour = side.get_side_contour()
                if contour:
                    x, y = contour[len(contour) // 2]
                    cv2.putText(img, str(side.get_side_info()), (x, y), font, 1, (255, 0, 255), 4, cv2.LINE_AA)

            cv2.imshow(f'Piece {piece.get_index()}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    pieces = myPuzzle.get_pieces()
    for piece in pieces:
        if piece.sides:
            type_of_sides(piece, windows)

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Sides info found...")
    return ()

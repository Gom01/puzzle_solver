import cv2 as cv

from Processing_puzzle.side import Side

def find_sides(myPuzzle, window=False):

    pieces = myPuzzle.get_pieces()
    for i, piece in enumerate(pieces):
        corners = piece.get_corners()

        if corners[0] != [-1, -1]:

            side1, side2, side3, side4 = [], [], [], []

            corners = corners[:]

            contour = piece.get_contours()

            # Trouver le point de départ qui correspond au premier coin
            start_index = contour.index(corners[0])
            current_side = side1
            visited_corners = [corners[0]]

            # Initialisation de la première variable point
            point = contour[start_index]
            point_origin = point
            index = start_index

            # On commence à remplir les côtés en suivant le contour
            while point != point_origin or len(visited_corners) < 4:
                point = contour[index]

                 # Vérifier si le point est un coin
                if point in corners and point not in visited_corners:
                    visited_corners.append(point)

                    # Changer de côté lorsque nous rencontrons un nouveau coin
                    if len(visited_corners) == 2:
                        current_side = side2
                    elif len(visited_corners) == 3:
                        current_side = side3
                    elif len(visited_corners) == 4:
                        current_side = side4

                # Ajouter le point au côté actuel
                current_side.append(point)

                # Avancer dans le contour
                index = (index + 1) % len(contour)

            if window:
                img = piece.get_color_image()
                corner1, corner2, corner3, corner4 = corners
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

                # Dessiner chaque côté
                for i, side in enumerate([side1, side2, side3, side4]):
                    color = colors[i]
                    for j in range(len(side) - 1):
                        cv.line(img, side[j], side[j + 1], color, 4)

                # Dessiner chaque coin (cercles)
                for i, corner in enumerate([corner1, corner2, corner3, corner4]):
                    color = (255, 90, 0)  # Jaune
                    # Dessiner un cercle à chaque coin
                    cv.circle(img, corner, radius=5, color=color, thickness=-1)

                cv.imshow("4 Sides", img)
                cv.waitKey(0)
                cv.destroyAllWindows()


            piece.set_sides(side1, side2, side3, side4)
        else:
            print("No sides found for this piece ")
            continue


    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Sides found...")
    return

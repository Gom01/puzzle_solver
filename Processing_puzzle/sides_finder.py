import cv2 as cv

from Processing_puzzle.Side import Side


def find_sides(myPuzzle):

    def find_4_sides(piece, list_corners, display=False, time=0):
        def check_corners(corner):
            x, y = corner
            return x != -1 and y != -1

        def display_4_sides(img, sides, corners, window, time):
            side1, side2, side3, side4 = sides
            corner1, corner2, corner3, corner4 = corners
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Bleu, Vert, Rouge, Jaune

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

            if window:
                cv.imshow("4 Sides", img)
                cv.waitKey(time)
                cv.destroyAllWindows()

        # Vérifier que les coins sont valides
        if all(check_corners(corner) for corner in list_corners):
            # Initialisation des côtés
            side1, side2, side3, side4 = [], [], [], []

            # Définition de l'ordre des coins, pour éviter toute confusion
            corners = list_corners[:]
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

            # Afficher les côtés et les coins si nécessaire
            display_4_sides(piece.get_color_image(), (side1, side2, side3, side4), list_corners, display, time)


            return side1, side2, side3, side4
        else:
            return None, None, None, None


    pieces = myPuzzle.get_pieces()
    for i, piece in enumerate(pieces):
        cnt = piece.get_contours()
        corners = piece.get_corners()
        side1, side2, side3, side4 = find_4_sides(piece, piece.get_corners(), False, 0)

        print(side1)

        side1_info = Side()
        side1_info.set_side_points(side1)

        side2_info = Side()
        side2_info.set_side_points(side2)

        side3_info = Side()
        side3_info.set_side_points(side3)

        side4_info = Side()
        side4_info.set_side_points(side4)

        print(side1_info.get_side_points())

        name = f"Pièce {i}"

        piece.set_name(name)

        name_piece = piece.get_name()

        print(name_piece)

        side1_info.set_side_name(f"{name_piece} : side 1")
        side2_info.set_side_name(f"{name_piece} : side 2")
        side3_info.set_side_name(f"{name_piece} : side 3")
        side4_info.set_side_name(f"{name_piece} : side 4")


        piece.set_sides(side1_info,side2_info,side3_info,side4_info)


    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("All sides saved ! ")
    return()


















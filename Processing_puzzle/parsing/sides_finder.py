import cv2 as cv

from Processing_puzzle.side import Side

def find_sides(myPuzzle):
    def find_4_sides(piece, list_corners, display=False, time=0):
        def check_corners(corner):
            x, y = corner
            return x != -1 and y != -1

        def display_4_sides(img_c, sides, corners, window, time):
            img = img_c.copy()
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
            side1_color, side2_color, side3_color, side4_color = [], [], [], []

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

                # Ajouter la couleur au côté
                if current_side == side1:
                    side1_color.append(piece.get_color_contour()[index])
                elif current_side == side2:
                    side2_color.append(piece.get_color_contour()[index])
                elif current_side == side3:
                    side3_color.append(piece.get_color_contour()[index])
                elif current_side == side4:
                    side4_color.append(piece.get_color_contour()[index])

                # Avancer dans le contour
                index = (index + 1) % len(contour)

            # Afficher les côtés et les coins si nécessaire
            display_4_sides(piece.get_color_image(), (side1, side2, side3, side4), list_corners, display, time)

            # Now you can create Side objects and set colors
            side1_info = Side()
            side1_info.set_side_points(side1)
            side1_info.set_side_colors(side1_color)

            side2_info = Side()
            side2_info.set_side_points(side2)
            side2_info.set_side_colors(side2_color)

            side3_info = Side()
            side3_info.set_side_points(side3)
            side3_info.set_side_colors(side3_color)

            side4_info = Side()
            side4_info.set_side_points(side4)
            side4_info.set_side_colors(side4_color)

            return side1_info, side2_info, side3_info, side4_info
        else:
            return None, None, None, None

    pieces = myPuzzle.get_pieces()
    for i, piece in enumerate(pieces):
        corners = piece.get_corners()
        side1_info, side2_info, side3_info, side4_info = find_4_sides(piece, piece.get_corners(), False, 0)

        name = f"Pièce {i}"
        piece.set_name(name)

        side1_info.set_side_name(f"{name} : side 1")
        side2_info.set_side_name(f"{name} : side 2")
        side3_info.set_side_name(f"{name} : side 3")
        side4_info.set_side_name(f"{name} : side 4")

        piece.set_sides(side1_info, side2_info, side3_info, side4_info)

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Sides found...")
    return

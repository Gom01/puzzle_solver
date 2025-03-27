from Processing_puzzle import Puzzle as p
import cv2 as cv
import numpy as np


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



myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()

for piece in pieces:

    cnt = piece.get_contours()
    corners = piece.get_corners()
    print("Corners :",corners)
    side1, side2, side3, side4 = find_4_sides(piece, piece.get_corners(),True,0)
    print("Side 1 :",side1)


    #cv2.imshow("img",piece.get_color_image())
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()











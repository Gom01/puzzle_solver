from Processing_puzzle import Puzzle as p
import cv2
import numpy as np

def get_next_point(piece):
    """
    Récupère le point suivant dans le contour (parcours circulaire).

    :return: Tuple (x, y) du point suivant.
    """
    if not hasattr(piece, 'index'):  # Vérifie si l'attribut 'index' existe
        piece.index = 0  # Initialisation de l'attribut 'index' si nécessaire

    if len(piece.get_contours()) == 0:  # Vérifie si les contours sont vides
        print("Erreur: Aucun contour trouvé.")
        return None

    contour = piece.get_contours()[0]  # On suppose qu'il y a au moins un contour
    if len(contour) == 0:  # Vérifie si le contour est vide
        print("Erreur: Le contour est vide.")
        return None

    # Assure-toi que l'index reste dans la plage du contour
    piece.index = (piece.index + 1) % len(contour)


    ##TODO ??? CONTOUR ? tuple(contour[self.index][0])
    return tuple(contour)  # Retourne (x, y)


def find_4_sides(piece, list_corners):
    if (len(list_corners) == 4):

        list_corners = list_corners

        corner_1 = list_corners[0]
        corner_2 = list_corners[1]
        corner_3 = list_corners[2]
        corner_4 = list_corners[3]

        # Initialisation des côtés
        side1 = []
        side2 = []
        side3 = []
        side4 = []

        # Définition de l'ordre des coins, pour éviter toute confusion lors de l'itération
        corners = [corner_1, corner_2, corner_3, corner_4]

        # Point initial du contour
        point_origin = get_next_point(piece)

        while point_origin != corner_1:
            point_origin = get_next_point(piece)

        # print(point_origin," - ",corner_1)

        point_origin = get_next_point(piece)

        # print(point_origin," - ",corner_1)

        point = point_origin

        # Variable pour suivre quel côté nous sommes en train de remplir
        current_side = side1
        current_corner_index = 0

        # Liste des coins que nous avons rencontrés pour la première fois
        visited_corners = []
        visited_corners.append(corner_1)
        corners.remove(corner_1)

        # Tant qu'on n'a pas parcouru tout le contour
        while point != point_origin or len(visited_corners) < 4:
            # Récupérer le prochain point sur le contour
            point = get_next_point(piece)

            # Vérifie si le point est un coin
            if point in corners:
                # Si nous passons d'un coin à un autre, on commence un nouveau côté
                if point not in visited_corners:
                    visited_corners.append(point)
                    current_corner_index += 1

                    # On change de côté chaque fois qu'on passe à un coin
                    if current_corner_index == 1:
                        current_side = side2
                    elif current_corner_index == 2:
                        current_side = side3
                    elif current_corner_index == 3:
                        current_side = side4

            # Ajoute le point au côté actuel
            current_side.append(point)

            piece.side_1 = side1
            piece.side_2 = side2
            piece.side_3 = side3
            piece.side_4 = side4
            return(side1, side2, side3, side4)

myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()

for idx, piece in enumerate(pieces):
    side1, side2, side3, side4 = find_4_sides(piece, piece.get_corners())
    print(side1)










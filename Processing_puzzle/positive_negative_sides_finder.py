import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from Processing_puzzle import Puzzle as p


def drawing_sides(piece):
    sides = piece.get_4_sides()

    # Centre de masse du contour global de la pièce
    piece_contour_R = np.array(piece.get_contours()).reshape((-1, 1, 2))
    M1 = cv.moments(piece_contour_R)

    type = piece.get_4_sides_info()[3]

    points = sides[3]
    print(points)

    points = np.array(points, dtype=np.int32).reshape((-1, 2))

    picture = piece.get_black_white_image()
    cv2.drawContours(picture, [points], 0, (255, 0, 255), 2)
    cv2.imshow('picture', picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cx = int(M1["m10"] / M1["m00"])
    cy = int(M1["m01"] / M1["m00"])

    # Étape 1 : Translation pour ramener le premier point à (0,0)
    x0, y0 = points[0]
    translated_points = [(x - x0, y - y0) for x, y in points]

    # Nouveau dernier point après translation
    x1_t, y1_t = translated_points[-1]

    # Étape 2 : Calcul de l'angle de rotation
    theta = np.arctan2(y1_t, x1_t)  # Angle entre le premier et dernier point

    # Matrice de rotation
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Application de la rotation
    rotated_points = [tuple(np.dot(rotation_matrix, [x, y])) for x, y in translated_points]

    # Transformation des coordonnées du centre de masse (cx, cy)
    transformed_cx, transformed_cy = np.dot(rotation_matrix, [cx - x0, cy - y0])

    # Si la pièce est concave, appliquer une rotation finale de 180 degrés
    if type == "concave":
        # Matrice de rotation de 180 degrés
        rotation_180 = np.array([[-1, 0], [0, -1]])

        # Appliquer la rotation de 180 degrés aux points transformés
        rotated_points = [tuple(np.dot(rotation_180, [x, y])) for x, y in rotated_points]

        # Transformation du centre de masse
        transformed_cx, transformed_cy = np.dot(rotation_180, [transformed_cx, transformed_cy])

    # Séparation des coordonnées X et Y
    x_values, y_values = zip(*rotated_points)

    # Création de la courbe
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5, label="Points transformés")

    # Ajout du centre de masse transformé
    plt.scatter(transformed_cx, transformed_cy, color='r', label=f'Centre de masse ({transformed_cx:.2f}, {transformed_cy:.2f})', zorder=5)

    plt.axhline(0, color='black', linewidth=1)  # Ligne horizontale
    plt.axvline(0, color='black', linewidth=1)  # Ligne verticale
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Courbe avec points alignés et centre de masse")
    plt.xlabel("X aligné")
    plt.ylabel("Y ajusté")
    plt.legend()

    # Affichage
    plt.show()


# Charger et traiter le puzzle
myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()

# Appliquer la fonction pour chaque pièce
this_piece = pieces[7]

drawing_sides(this_piece)

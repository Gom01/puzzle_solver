import cv2
import cv2 as cv
import numpy as np
from Processing_puzzle import Puzzle as p


import cv2 as cv
import numpy as np
from Processing_puzzle import Puzzle as p



def type_of_sides(piece):
    sides = piece.get_4_sides()  # Liste des côtés
    img = piece.get_black_white_image()  # Image en noir et blanc

    # Centre de masse du contour global de la pièce
    piece_contour = np.array(piece.get_contours()).reshape((-1, 1, 2))
    M1 = cv.moments(piece_contour)
    if M1["m00"] != 0:
        x_M1 = int(M1["m10"] / M1["m00"])
        y_M1 = int(M1["m01"] / M1["m00"])
    else:
        x_M1, y_M1 = 0, 0  # Éviter division par zéro

    for i, side in enumerate(sides):
        # Transformer le côté en un contour OpenCV
        side_array = np.array(side).reshape((-1, 1, 2))

        # Enveloppe convexe du côté
        hull = cv.convexHull(side_array)

        # Calcul des moments du **hull**
        M = cv.moments(hull)
        if M["m00"] != 0:
            x_M = int(M["m10"] / M["m00"])
            y_M = int(M["m01"] / M["m00"])
        else:
            x_M, y_M = 0, 0  # Éviter division par zéro

        # Déterminer les points extrêmes du côté
        x1, y1 = side_array[0][0]   # Premier point (extrême gauche)
        x2, y2 = side_array[-1][0]  # Dernier point (extrême droit)

        # Calcul de l'équation de la droite entre les extrémités
        if x2 - x1 != 0:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
        else:
            a = None  # Droite verticale

        # Vérifier la position relative du centre de masse du hull
        def position_relative(x, y):
            if a is not None:
                return y - (a * x + b)  # Positif si au-dessus, négatif si en dessous
            else:
                return x - x1  # Pour une droite verticale

        pos_M1 = position_relative(x_M1, y_M1)
        pos_M = position_relative(x_M, y_M)

        print("Area :",cv2.contourArea(hull))

        # Déterminer la nature du côté
        if cv.contourArea(hull) < 1000:
            type_cote = "droit"
            color = (255, 255, 0)  # Jaune
        elif np.sign(pos_M1) == np.sign(pos_M):
            type_cote = "concave"
            color = (255, 0, 0)  # Bleu
        else:
            type_cote = "convexe"
            color = (0, 255, 0)  # Vert

        print(f"Le côté {i} est {type_cote}.")

        # Dessiner les résultats
        img_copy = img.copy()
        cv.drawContours(img_copy, [hull], -1, (0, 255, 0), thickness=2)  # Hull en vert
        cv.circle(img_copy, (x_M1, y_M1), 5, (255, 0, 0), -1)  # Centre de masse du puzzle (bleu)
        cv.circle(img_copy, (x_M, y_M), 5, (0, 0, 255), -1)  # Centre de masse du hull (rouge)
        cv.circle(img_copy, (x1, y1), 5, (255, 255, 255), -1)  # Points extrêmes en blanc
        cv.circle(img_copy, (x2, y2), 5, (255, 255, 255), -1)
        cv.line(img_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Afficher l'image
        cv.imshow(f"Côté {i}", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

        print(f"Le côté {i} a été traité.")

    '''
    corners = piece.get_corners()
    corners = corners[:]
    sides = piece.get_4_sides()

    # Créer une image vide (noire)
    height, width = piece.get_black_white_image().shape[:2]
    image_binary = np.zeros((height, width), dtype=np.uint8)

    # Dessiner le contour de side[0] sur l'image binaire
    points = np.array(sides[0], dtype=np.int32)  # Convertir la liste de coordonnées en array numpy
    cv2.polylines(image_binary, [points], isClosed=False, color=255, thickness=1)  # Tracer les contours


    img = piece.get_black_white_image()
    # Appliquer la transformée de Hough sur l'image binaire
    lines = cv2.HoughLinesP(image_binary, 100, np.pi / 180, threshold=20, minLineLength=90, maxLineGap=9000000)

    # Dessiner les lignes détectées
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Afficher l'image avec les lignes détectées
    cv2.imshow("Lignes détectées", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    '''


# Charger et traiter le puzzle
myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()

# Appliquer la fonction pour chaque pièce
for piece in pieces:
    type_of_sides(piece)

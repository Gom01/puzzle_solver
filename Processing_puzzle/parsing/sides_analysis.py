import cv2
import cv2 as cv
import numpy as np
from Processing_puzzle import Puzzle as p


import cv2 as cv
import numpy as np
from Processing_puzzle import Puzzle as p

def sides_information(myPuzzle):

    def type_of_sides(piece,windows=False):
        sides = piece.get_sides()  # Liste des côtés
        img = piece.get_color_image().copy() # Image en noir et blanc


        # Centre de masse du contour global de la pièce
        piece_contour = np.array(piece.get_contours()).reshape((-1, 1, 2))
        M1 = cv.moments(piece_contour)

        piece.set_moment(M1)

        if M1["m00"] != 0:
            x_M1 = int(M1["m10"] / M1["m00"])
            y_M1 = int(M1["m01"] / M1["m00"])
        else:
            x_M1, y_M1 = 0, 0  # Éviter division par zéro

        sides_info = []

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

            longueur = ((x2 - x1)**2 + (y2 - y1)**2)**(1 / 2)

            area_mini_rectangle = longueur
            #print(f"L'aire du rectangle est : {area_mini_rectangle} pixels²")

            if windows:
                print("Area :", cv2.contourArea(hull))

            # Déterminer la nature du côté
            if cv.contourArea(hull) < area_mini_rectangle*5:
                type_cote = "droit"
                color = (255, 255, 0)  # Jaune
                sides_info.append(0)
            elif np.sign(pos_M1) == np.sign(pos_M):
                type_cote = "concave"
                color = (255, 0, 0)  # Bleu
                sides_info.append(-1)
            else:
                type_cote = "convexe"
                color = (0, 255, 0)  # Vert
                sides_info.append(1)

            #if windows:
               # print(f"Le côté {i} est {type_cote}.")

            # Dessiner les résultats
            cv.drawContours(img, [hull], -1, (0, 255, 0), thickness=2)  # Hull en vert
            cv.circle(img, (x_M1, y_M1), 5, (255, 0, 0), -1)  # Centre de masse du puzzle (bleu)
            cv.circle(img, (x_M, y_M), 5, (0, 0, 255), -1)  # Centre de masse du hull (rouge)
            cv.circle(img, (x1, y1), 5, (255, 255, 255), -1)  # Points extrêmes en blanc
            cv.circle(img, (x2, y2), 5, (255, 255, 255), -1)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if(windows):
            print(sides_info)
            # Afficher l'image
            cv.imshow(f"Côté {i}", img)
            cv.waitKey(0)
            cv.destroyAllWindows()

            #print(f"Le côté {i} a été traité.")



        return sides_info


    pieces = myPuzzle.get_pieces()

    # Appliquer la fonction pour chaque pièce
    for piece in pieces:
        if piece.sides != []:
            sides_info = type_of_sides(piece,False)
            x,y,z,w = sides_info
            piece.set_sides_info(x,y,z,w)


    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Sides info found...")
    return ()



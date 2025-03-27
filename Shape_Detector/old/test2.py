import cv2 as cv
import numpy as np

from Puzzle import Puzzle

# Créer une instance de Puzzle
puzzle = Puzzle()

# Charger le puzzle depuis le fichier pickle
puzzle.load_puzzle('mon_puzzle.pickle')

# Vérifier que les pièces sont chargées correctement
pieces = puzzle.get_pieces()

# Afficher ou traiter les pièces du puzzle
for idx, piece in enumerate(pieces):
    print(f"Pièce {idx + 1} - Position: {piece.get_position()}")

    # Récupérer l'image couleur
    img_color = piece.get_image()




    # Récupérer les contours
    contours = piece.get_contours()

    piece.animate()


    # Créer une image noire de la même taille que l'image d'origine
    black_image = np.zeros_like(img_color)

    # Dessiner les contours en blanc sur l'image noire
    cv.drawContours(black_image, contours, -1, (255, 255, 255), 2)

    # Afficher l'image avec les contours dessinés sur fond noir
    cv.imshow("Contours sur fond noir", black_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Convertir l'image noire en niveaux de gris pour la détection de coins
    gray = cv.cvtColor(black_image, cv.COLOR_BGR2GRAY)

    # Détection des coins avec Harris
    dst = cv.cornerHarris(gray, 10, 7, 0.06)

    # Amélioration de la visibilité des coins
    dst = cv.dilate(dst, (7,7))

    # Seuillage pour obtenir des points bien définis
    _, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # Trouver les centres des coins détectés
    _, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # **Affichage des coins détectés AVANT cornerSubPix()**
    img_harris = piece.get_image_color()  # Copier l'image de la pièce pour l'affichage

    moments = piece.get_moments()
    x_moment = moments[0]
    y_moment = moments[1]

    cv.circle(img_harris, (int(x_moment), int(y_moment)), 5, (255, 255, 0), -1)



    for x, y in centroids:
        cv.circle(img_harris, (int(x), int(y)), 5, (0, 0, 255), -1)  # Rouge pour les coins détectés
        cv.line(img_harris, (int(x), int(y)), (int(x_moment), int(y_moment)), (0, 255, 0), 2)






    # Afficher les coins détectés sur l'image de la pièce
    cv.imshow(f'Coins Harris - Pièce {idx + 1}', img_harris)
    cv.waitKey(0)
    cv.destroyAllWindows()

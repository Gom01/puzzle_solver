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



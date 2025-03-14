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
    print(f"Piece {idx + 1} - Position: {piece.get_position()}")

    # Récupérer la position de la pièce
    shift_x, shift_y = piece.get_position()

    # Récupérer l'image recadrée
    img = piece.get_image()

    # Récupérer les contours et ajuster les coordonnées
    contours = piece.get_contours()


    # Dessiner les contours ajustés sur l'image recadrée
    cv.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Afficher l'image avec les contours ajustés
    cv.imshow(f'Piece {idx + 1}', img)
    cv.waitKey(0)

cv.destroyAllWindows()

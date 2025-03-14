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

    # Afficher l'image de la pièce (par exemple)
    cv.imshow(f'Piece {idx + 1}', piece.get_image())
    cv.waitKey(0)

cv.destroyAllWindows()
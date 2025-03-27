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
    image = piece.get_image()

    # Convertir en niveaux de gris
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours
    contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialisation de la liste des coins détectés
    detected_corners = []

    # Parcours de chaque contour
    for contour in contours:
        # Calcul des moments pour chaque contour
        M = cv.moments(contour)

        # Vérification des moments (on évite la division par zéro)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Calcul des Moments de Hu
        hu_moments = cv.HuMoments(M).flatten()

        # Critère pour la détection des coins basé sur les Moments de Hu
        # Vous pouvez ajuster ce critère selon vos besoins
        if np.abs(hu_moments[0]) > 0.001:  # Valeur empirique pour filtrer les coins
            detected_corners.append((cX, cY))

    # Afficher les coins détectés
    for corner in detected_corners:
        x, y = corner
        cv.circle(image, (x, y), 5, (0, 0, 255), -1)  # Marquer les coins en rouge

    # Afficher l'image avec les coins détectés
    cv.imshow(f'Coins détectés - Pièce {idx + 1}', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

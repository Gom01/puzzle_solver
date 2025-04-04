import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev

from Processing_puzzle import Puzzle as p

from scipy.signal import correlate
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.stats import pearsonr

# Charger et traiter le puzzle
myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()

piece_scores = []


for piece1 in pieces:



    for piece2 in pieces:



        for side1, side2, eq1, eq2 in compatible_sides:
            # Calculer la corrélation entre les équations des côtés
            corr_x, _ = pearsonr(eq1[0], eq2[0])
            corr_y, _ = pearsonr(eq1[1], eq2[1])

            # Calculer la corrélation moyenne
            avg_corr = (corr_x + corr_y) / 2

            # Ajouter le score de la pièce (avec la corrélation) à la liste
            piece_scores.append((piece, side1, side2, avg_corr))

# Trier les pièces par la corrélation maximale (du meilleur au pire)
sorted_piece_scores = sorted(piece_scores, key=lambda x: x[3], reverse=True)

# Afficher le classement
for rank, (piece, side1, side2, score) in enumerate(sorted_piece_scores, 1):
    print(f"Rank {rank}: Piece {piece}, Side1: {side1}, Side2: {side2}, Score: {score}")



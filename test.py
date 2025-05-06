import numpy as np
from math import sqrt


def color_similarities(colors1, colors2, weight=1.0):
    """
    Compare deux listes de couleurs et retourne un score de similarité basé sur la distance euclidienne.

    Chaque couleur est un tableau NumPy [R, G, B].
    """
    score = 0
    length = min(len(colors1), len(colors2))  # On s'assure de comparer jusqu'à la longueur minimale des deux listes

    for c1, c2 in zip(colors1[:length], colors2[:length]):
        # Distance euclidienne entre les deux couleurs (R, G, B)
        dist = sqrt(np.sum((c1 - c2) ** 2))  # Racine carrée de la somme des carrés des différences
        similarity = int(weight * (255 * sqrt(3)) / (dist + 1))  # On inverse la distance pour la similarité
        score += similarity

    return score


# Exemple de données de couleurs dominantes
piece0_side0 = [
    np.array([85, 94, 110]), np.array([79, 91, 106]), np.array([88, 96, 111]),
    np.array([89, 92, 107]), np.array([115, 115, 119]), np.array([121, 125, 133]),
    np.array([130, 132, 141]), np.array([72, 67, 88]), np.array([72, 66, 84]),
    np.array([72, 68, 87]), np.array([55, 57, 79]), np.array([69, 71, 83]),
    np.array([73, 69, 79]), np.array([67, 75, 84]), np.array([90, 99, 106]),
    np.array([89, 101, 108]), np.array([93, 103, 115]), np.array([73, 78, 90]),
    np.array([68, 61, 74]), np.array([68, 59, 70])
]

piece0_side1 = [
    np.array([113, 110, 118]), np.array([98, 106, 116]), np.array([102, 107, 111]),
    np.array([109, 120, 118]), np.array([130, 136, 131]), np.array([115, 120, 116]),
    np.array([121, 130, 125]), np.array([56, 60, 70]), np.array([50, 51, 60]),
    np.array([118, 121, 130]), np.array([106, 104, 116]), np.array([96, 98, 108]),
    np.array([119, 134, 137]), np.array([137, 146, 150]), np.array([131, 140, 148]),
    np.array([139, 144, 150]), np.array([139, 150, 162]), np.array([140, 157, 170]),
    np.array([138, 155, 162]), np.array([130, 146, 152])
]

# Calcul du score de similarité
score = color_similarities(piece0_side0, piece0_side1, weight=1.0)
print("Score de similarité des couleurs:", score)
from Processing_puzzle import Puzzle as p

import cv2
import numpy as np
from scipy.interpolate import splev
from scipy.stats import pearsonr
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from Processing_puzzle.Tools import Tools


# Fonction pour convertir le plot de Matplotlib en image OpenCV
def plot_to_image(fig):
    # Utiliser un canvas pour rendre la figure dans un format bitmap
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convertir le bitmap en un tableau numpy
    img_array = np.array(canvas.renderer.buffer_rgba())

    # Convertir RGBA en BGR pour OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_bgr


# Fonction d'affichage des courbes dans OpenCV
def afficher_comparaison_courbes(tck1, tck2, score, label1, label2):
    u_common = np.linspace(0, 1, 100)
    x1, y1 = splev(u_common, tck1)
    x2, y2 = splev(u_common, tck2)

    # Créer la figure dans Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x1, y1, label=f'{label1}', color='blue')
    ax.plot(x2, y2, label=f'{label2}', color='red', linestyle='--')
    ax.set_title(f'Comparaison des courbes - Corrélation moyenne: {score:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    # Convertir la figure en image OpenCV
    plot_image = plot_to_image(fig)

    return plot_image  # Retourner l'image pour l'affichage dans cv2.imshow


def afficher_comparaison_pieces(piece1, side1, piece2, side2, tck1, tck2, score, label1, label2):
    points1 = np.array(side1.get_side_points(), dtype=np.int32).reshape((-1, 2))
    points2 = np.array(side2.get_side_points(), dtype=np.int32).reshape((-1, 2))

    picture1 = piece1.get_color_image().copy()
    picture2 = piece2.get_color_image().copy()

    side1_inf = side1.get_side_info()
    side2_inf = side2.get_side_info()

    # Dessiner les contours sur les images
    cv2.drawContours(picture1, [points1], 0, (255, 255, 255), 5)
    cv2.drawContours(picture2, [points2], 0, (255, 255, 255), 5)

    # Calculer l'image des courbes
    plot_image = afficher_comparaison_courbes(tck1, tck2, score, label1, label2)
    plot_image_resized = cv2.resize(plot_image, (240, 240))

    # Récupération des rotations
    rotation_1 = side1.get_rotation()
    rotation_2 = side2.get_rotation()

    tools = Tools()

    rotated_picture1 = tools.rotate_image(picture1, rotation_1, piece1.get_moment())
    rotated_picture2 = tools.rotate_image(picture2, rotation_2, piece2.get_moment())

    # Redimensionner les images
    target_size = (240, 240)
    rotated_picture1 = cv2.resize(rotated_picture1, target_size)
    rotated_picture2 = cv2.resize(rotated_picture2, target_size)

    # Déterminer l’ordre : concave en haut
    if side1_inf == "convexe":
        pile_pieces = np.vstack((rotated_picture1, rotated_picture2))
    else:
        pile_pieces = np.vstack((rotated_picture2, rotated_picture1))

    # Empiler verticalement la pile de pièces avec l'image des courbes
    final_image = np.vstack((pile_pieces, plot_image_resized))

    # Affichage final
    cv2.imshow("Comparaison des Courbes et Pièces (Verticale)", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Charger et traiter le puzzle
myPuzzle = p.Puzzle()
myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
pieces = myPuzzle.get_pieces()

piece_scores = []
pieces_analysis = pieces.copy()

for piece1 in pieces_analysis:  # Copier la liste pour éviter modification durant l'itération

    sides1 = piece1.get_sides()

    for side1 in sides1:
        for piece2 in pieces_analysis:
            sides2 = piece2.get_sides()

            for side2 in sides2:
                side1_info = side1.get_side_info()
                side1_eq = side1.get_side_eq()  # (tck, u)

                side2_info = side2.get_side_info()
                side2_eq = side2.get_side_eq()  # (tck, u)

                if (side1_info, side2_info) in [("concave", "convexe"), ("convexe", "concave")]:
                    # Évaluer les splines pour obtenir les coordonnées
                    tck1, u1 = side1_eq
                    tck2, u2 = side2_eq

                    # Rééchantillonnage sur une grille commune (par ex. 100 points)
                    u_common = np.linspace(0, 1, 100)
                    x1, y1 = splev(u_common, tck1)
                    x2, y2 = splev(u_common, tck2)

                    # Vérifier la présence de NaN dans les coordonnées
                    if np.any(np.isnan(x1)) or np.any(np.isnan(y1)) or np.any(np.isnan(x2)) or np.any(np.isnan(y2)):
                        continue

                    # Calcul de la corrélation
                    try:
                        corr_x, _ = pearsonr(x1, x2)
                        corr_y, _ = pearsonr(y1, y2)
                        avg_corr = (corr_x + corr_y) / 2

                        # Ajouter le score à la liste
                        piece_scores.append(((piece1, side1, side1_info), (piece2, side2, side2_info), avg_corr))
                    except Exception as e:
                        print(f"Erreur de corrélation entre pièces {piece1} et {piece2} : {e}")

# Trier les pièces par corrélation décroissante
sorted_piece_scores = sorted(piece_scores, key=lambda x: x[2], reverse=True)

# Afficher les 5 meilleures correspondances
print("\n--- Affichage des 5 meilleures paires ---")
top_n = 30
for i in range(min(top_n, len(sorted_piece_scores))):
    (piece1, side1, info1), (piece2, side2, info2), score = sorted_piece_scores[i]
    tck1, _ = side1.get_side_eq()
    tck2, _ = side2.get_side_eq()
    label1 = f"Pièce {id(piece1)} ({info1})"
    label2 = f"Pièce {id(piece2)} ({info2})"


    # Afficher les pièces et les courbes dans une seule fenêtre
    afficher_comparaison_pieces(piece1, side1, piece2, side2, tck1, tck2, score, label1, label2)


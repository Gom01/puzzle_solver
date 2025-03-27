import os
from operator import truediv

import cv2 as cv
import numpy as np


from Piece_Puzzle import Piece_Puzzle
from Puzzle import Puzzle
from Tools import Tools  # Assurez-vous que Tools est bien importé depuis le bon fichier

WINDOW_SIZE = (1000, 800)
WINDOW_SIZE_PIECE = (200, 200)
X_PERCENT = 50  # Ajustable (ex : 30% autour de la médiane)
MARGIN_PERCENT = 1  # Ajuste la taille de la marge autour des contours

image_path = '../../images/p1_b/Natel.Black1.jpg'

tools = Tools()


puzzle = Puzzle()


# Charger l'image
image = cv.imread(image_path)
if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")


# Convertir en niveaux de gris
im_gray = tools.convert_to_grayscale(image, window=False, window_size=WINDOW_SIZE, time=0)

# Appliquer le seuillage Otsu
im_thresh = tools.apply_otsu_threshold(im_gray, window=False, window_size=WINDOW_SIZE, time=0)

# Suppression du bruit
im_clean = tools.remove_noise(im_thresh, kernel_size=(3, 3), iterations=20, window=False, window_size=WINDOW_SIZE, time=0)

# Trouver les contours
contours,_ = tools.find_contours(image,im_clean, window=False, window_size=WINDOW_SIZE, time=0)

im_with_black_background,contours_filtered,min_valid_area,max_valid_area  = tools.filter_contours_by_area(image,contours, 1800, window=False, window_size=WINDOW_SIZE, time=0)

# Stocker les contours des pièces individuelles
assembled_pieces = []
all_piece_contours = []

# Boucle de traitement pièces
for contour_piece in contours_filtered:
    x, y, w, h, piece_image = tools.get_bounding_rect(im_with_black_background, contour_piece, 10, False, WINDOW_SIZE_PIECE, 1000)

    piece_image_gray = tools.convert_to_grayscale(piece_image, window=False, window_size=WINDOW_SIZE_PIECE, time=0)

    piece_image_thresh = tools.apply_otsu_threshold(piece_image_gray, window=False, window_size=WINDOW_SIZE_PIECE, time=0)

    im_piece_clean = tools.remove_noise(piece_image_thresh, kernel_size=(3, 3), iterations=2, window=False, window_size=WINDOW_SIZE_PIECE, time=0)

    # Détection des contours
    contours_piece, _ = cv.findContours(im_piece_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    im_piece_contours = piece_image.copy()
    cv.drawContours(im_piece_contours, contours_piece, -1, (0, 255, 0), 2)

    # Filtrer les contours des pièces avec la médiane
    contours_filtered_p = [cnt_p2 for cnt_p2 in contours_piece if min_valid_area <= cv.contourArea(cnt_p2) <= max_valid_area]

    # Appliquer le lissage sur les contours filtrés
    smoothed_contours = []
    for cnt_p2 in contours_filtered_p:
        # Convertir les coordonnées du contour en tableau
        cnt_p2 = np.array(cnt_p2, dtype=np.float32)  # Nécessaire pour lissage
        smoothed_cnt = cv.GaussianBlur(cnt_p2, (3,3 ), 0)  # Appliquer le flou Gaussien
        smoothed_contours.append(smoothed_cnt)

    # Dessiner les contours lissés
    im_contours_filtered_p = piece_image.copy()
    for smoothed_cnt in smoothed_contours:
        cv.drawContours(im_contours_filtered_p, [smoothed_cnt.astype(int)], -1, (0, 255, 0), 2)

    # Créer un masque pour les contours filtrés de la pièce
    piece_mask = np.zeros(piece_image.shape[:2], dtype=np.uint8)
    for smoothed_cnt in smoothed_contours:
        cv.drawContours(piece_mask, [smoothed_cnt.astype(int)], -1, (255), thickness=cv.FILLED)

    # Appliquer le masque inversé pour transformer l'extérieur des contours en noir
    piece_with_black_background = cv.bitwise_and(piece_image, piece_image, mask=piece_mask)

    cv.imshow("Result _---",piece_mask)
    cv.imshow("Result _---",piece_image)
    cv.imshow("Result C_---",piece_with_black_background)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Remplacer l'intérieur par du blanc
    piece_with_black_background_with_white_inside = piece_with_black_background.copy()
    piece_with_black_background_with_white_inside[np.where(piece_mask == 255)] = [255, 255, 255]

    # Affichage combiné en une seule fenêtre
    resultat_final = np.hstack((
        cv.resize(cv.cvtColor(piece_image_gray, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(cv.cvtColor(piece_image_thresh, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(cv.cvtColor(im_piece_clean, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(im_piece_contours, WINDOW_SIZE_PIECE),
        cv.resize(im_contours_filtered_p, WINDOW_SIZE_PIECE),
        cv.resize(piece_with_black_background, WINDOW_SIZE_PIECE)
    ))

    cv.imshow(f'Contours de la pièce {x},{y}', resultat_final)
    cv.waitKey(500)
    cv.destroyAllWindows()

    # Créer l'objet Piece_Puzzle avec toutes les informations
    piece = Piece_Puzzle(piece_with_black_background_with_white_inside, piece_with_black_background, contours_filtered_p, x, y)

    puzzle.add_pieceset(piece)

    # Ajouter les contours globaux
    assembled_pieces.append(im_contours_filtered_p)
    for cnt in contours_filtered_p:
        cnt += [x, y]
        all_piece_contours.append(cnt)

# Dessiner les contours globaux sélectionnés
im_all_contours = image.copy()
cv.drawContours(im_all_contours, all_piece_contours, -1, (0, 255, 255), 10)

# Afficher le résultat final
cv.imshow('Image all', cv.resize(image, WINDOW_SIZE))
cv.imshow('Contours filtrés', cv.resize(im_with_black_background, WINDOW_SIZE))
cv.imshow('Contours sélectionnés selon X% de la médiane', cv.resize(im_all_contours, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Sauvegarder le puzzle dans un fichier pickle
puzzle.save_puzzle('mon_puzzle.pickle')
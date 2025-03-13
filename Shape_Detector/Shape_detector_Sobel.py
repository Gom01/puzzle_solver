import os
import cv2 as cv
import numpy as np

WINDOW_SIZE = (1000, 800)
WINDOW_SIZE_PIECE = (200, 200)
X_PERCENT = 50  # Ajustable (ex : 30% autour de la médiane)
MARGIN_PERCENT = 1  # Ajuste la taille de la marge autour des contours

# Charger l'image
#image_path = '../images/p1_b/Natel.Black1.jpg' # Bon
image_path = '../images/p1_b/Natel.Black.jpg' # Bon
#image_path = '../images/p1_b/Natel_Black_G.jpg' #Problème de résolution
#image_path = '../images/p1/WIN_20250306_15_09_28_Pro.jpg' #Pb du fond avec la table
#image_path = '../images/p2/WIN_20250306_15_11_32_Pro.jpg' #Pb du fond avec la table
assert os.path.exists(image_path), f"Erreur : le fichier {image_path} n'existe pas."
im = cv.imread(image_path)
assert im is not None, "Erreur : l'image ne peut pas être lue."

# Conversion en niveaux de gris
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# Seuil Otsu
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Suppression du bruit
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=20)
opening = cv.bitwise_not(opening)

# Détection des contours
contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Filtrer les contours avec une superficie min
areas = [cv.contourArea(cnt) for cnt in contours]
areas_filtered = [area for area in areas if area > 1800]

# Calcul de la médiane et définition des bornes selon X%
median_area = np.median(areas_filtered) if areas_filtered else 0
min_valid_area = median_area * (1 - X_PERCENT / 100)
max_valid_area = median_area * (1 + X_PERCENT / 100)

print(f"Aire médiane : {median_area}")
print(f"Intervalle de sélection : [{min_valid_area}, {max_valid_area}]")

# Filtrer les contours basés sur la médiane
contours_filtered = [cnt for cnt in contours if min_valid_area <= cv.contourArea(cnt) <= max_valid_area]

# Créer un masque pour remplir les contours
mask = np.zeros(im.shape[:2], dtype=np.uint8)

# Dessiner les contours sur le masque
cv.drawContours(mask, contours_filtered, -1, (255), thickness=cv.FILLED)

# Appliquer une dilatation pour créer une marge autour des contours
kernel = np.ones((5, 5), np.uint8)
mask_dilated = cv.dilate(mask, kernel, iterations=MARGIN_PERCENT)

# Inverser le masque dilaté pour avoir les zones extérieures en noir
mask_inv = cv.bitwise_not(mask_dilated)

# Appliquer le masque inversé pour transformer l'extérieur en noir
im_with_black_background = cv.bitwise_and(im, im, mask=mask_dilated)

# Afficher l'image avec le fond noir
cv.imshow('Image avec arrière-plan noir', cv.resize(im_with_black_background, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Stocker les contours des pièces individuelles
assembled_pieces = []
all_piece_contours = []

# Boucle de traitement des pièces
for contour in contours_filtered:
    x, y, w, h = cv.boundingRect(contour)
    piece_image = im_with_black_background[y:y + h, x:x + w]
    piece_gray = cv.cvtColor(piece_image, cv.COLOR_BGR2GRAY)

    # Seuil Otsu
    _, thresh_p = cv.threshold(piece_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Appliquer Sobel
    sobelx_piece = cv.Sobel(thresh_p, cv.CV_64F, 1, 0, ksize=3)
    sobely_piece = cv.Sobel(thresh_p, cv.CV_64F, 0, 1, ksize=3)
    sobel_piece = cv.magnitude(sobelx_piece, sobely_piece)
    sobel_piece = cv.convertScaleAbs(sobel_piece)

    # Suppression du bruit
    kernel_p = np.ones((3, 3), np.uint8)
    opening_p = cv.morphologyEx(thresh_p, cv.MORPH_OPEN, kernel, iterations=2)
    opening_p = cv.bitwise_not(opening_p)



    # Détection des contours
    contours_piece, _ = cv.findContours(opening_p, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    im_piece_contours = piece_image.copy()
    cv.drawContours(im_piece_contours, contours_piece, -1, (0, 255, 0), 2)

    # Filtrer contours des pièces avec la médiane
    contours_filtered_p = [cnt_p for cnt_p in contours_piece if min_valid_area <= cv.contourArea(cnt_p) <= max_valid_area]

    # Dessiner contours filtrés
    im_contours_filtered_p = piece_image.copy()
    cv.drawContours(im_contours_filtered_p, contours_filtered_p, -1, (0, 255, 0), 2)
    # Créer un masque pour les contours filtrés de la pièce
    piece_mask = np.zeros(piece_image.shape[:2], dtype=np.uint8)
    cv.drawContours(piece_mask, contours_filtered_p, -1, (255), thickness=cv.FILLED)

    # Appliquer le masque inversé pour transformer l'extérieur des contours en noir
    piece_with_black_background = cv.bitwise_and(piece_image, piece_image, mask=piece_mask)

    # Définir la taille de la marge (en pixels)
    marge_noir = 20  # Par exemple, marge de 20 pixels




    # Remplacement de l'intérieur par du blanc
    piece_with_black_background_with_white_inside = piece_with_black_background.copy()
    piece_with_black_background_with_white_inside[np.where(piece_mask == 255)] = [255, 255, 255]

    # Ajouter la marge noire autour de la pièce avec fond noir et intérieur blanc
    piece_with_margin = cv.copyMakeBorder(piece_with_black_background_with_white_inside,
                                          top=marge_noir,
                                          bottom=marge_noir,
                                          left=marge_noir,
                                          right=marge_noir,
                                          borderType=cv.BORDER_CONSTANT,
                                          value=[0, 0, 0])  # Marge noire

    pieces_separated = []  # Liste des images des pièces avec fond noir et intérieur blanc



    # Affichage combiné en une seule fenêtre
    resultat_final = np.hstack((
        cv.resize(cv.cvtColor(piece_gray, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(cv.cvtColor(thresh_p, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(cv.cvtColor(sobel_piece, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(cv.cvtColor(opening_p, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
        cv.resize(im_piece_contours, WINDOW_SIZE_PIECE),
        cv.resize(im_contours_filtered_p, WINDOW_SIZE_PIECE),
        cv.resize(piece_with_black_background, WINDOW_SIZE_PIECE)
        # Affichage avec fond noir
    ))

    # Définir le chemin du dossier
    output_folder = "list_pieces"

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



    output_path = os.path.join(output_folder, f"piece{x}_{y}.jpg")

    # Sauvegarder
    cv.imwrite(output_path, piece_with_margin)

    print(f"Image sauvegardée sous : {output_path}")

    cv.imshow(f'Contours de la pièce {x},{y}', resultat_final)
    cv.waitKey(1000)
    cv.destroyAllWindows()

    # Ajouter les contours globaux
    assembled_pieces.append(im_contours_filtered_p)
    for cnt in contours_filtered_p:
        cnt += [x, y]
        all_piece_contours.append(cnt)



# Dessiner les contours globaux sélectionnés
im_all_contours = im.copy()
cv.drawContours(im_all_contours, all_piece_contours, -1, (0, 255, 255), 10)

# Afficher le résultat final
cv.imshow('Image all', cv.resize(im, WINDOW_SIZE))
cv.imshow('Contours filtrés', cv.resize(im_with_black_background, WINDOW_SIZE))
cv.imshow('Contours sélectionnés selon X% de la médiane', cv.resize(im_all_contours, WINDOW_SIZE))
cv.waitKey(2000)
cv.destroyAllWindows()

# Redimensionner les images à la même taille
height = max(im.shape[0], im_with_black_background.shape[0], im_all_contours.shape[0])
width = max(im.shape[1], im_with_black_background.shape[1], im_all_contours.shape[1])

# Redimensionner les images pour qu'elles aient toutes la même taille
im_resized = cv.resize(im, (width, height))
im_with_black_background_resized = cv.resize(im_with_black_background, (width, height))
im_all_contours_resized = cv.resize(im_all_contours, (width, height))

# Combiner les images horizontalement
final_image = np.hstack((im_resized, im_with_black_background_resized, im_all_contours_resized))

# Afficher le résultat final
cv.imshow('Image all', cv.resize(final_image, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()
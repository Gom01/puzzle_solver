import os
import cv2 as cv
import numpy as np

WINDOW_SIZE = (900, 700)
WINDOW_SIZE_PIECE = (200, 200)

# Charger l'image
#image_path = '../images/p1_b/Natel.Black1.jpg'
image_path = '../images/p1_b/Natel.Black.jpg'

assert os.path.exists(image_path), f"Erreur : le fichier {image_path} n'existe pas."
im = cv.imread(image_path)
assert im is not None, "Erreur : l'image ne peut pas être lue."

# Conversion en niveaux de gris pour la détection des contours
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', cv.resize(gray, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Appliquer les filtres de Sobel pour les gradients
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # Dérivée en X
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # Dérivée en Y

# Magnitude du gradient
sobel = cv.magnitude(sobelx, sobely)

# Normalisation pour affichage
sobel = cv.convertScaleAbs(sobel)

cv.imshow('Sobel', cv.resize(sobel, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Seuillage automatique (Otsu) sur la magnitude du gradient
ret, th_sobel_otsu = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('Seuillage Sobel Otsu', cv.resize(th_sobel_otsu, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Détection des contours après Sobel avec Otsu
contours, _ = cv.findContours(th_sobel_otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
im_contours = im.copy()
cv.drawContours(im_contours, contours, -1, (0, 255, 0), 2)
cv.imshow('Contours détectés après Sobel Otsu', cv.resize(im_contours, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Calculer la superficie de chaque contour
areas = [cv.contourArea(cnt) for cnt in contours]

# Exclure les contours dont la superficie est inférieure à 900
areas_filtered = [area for area in areas if area >= 900]

# Calculer le quartile 25% des valeurs les plus hautes
highest_quartile_x = np.percentile(areas_filtered, 10)
print(f"Quartile % ..des valeurs les plus hautes (sans les surfaces < 900) : {highest_quartile_x}")

# Définir min_area basé sur le quartile 25% des valeurs les plus hautes
min_area = highest_quartile_x

# Filtrer les contours selon leur superficie
contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

# Dessiner les contours filtrés
im_contours_filtered = im.copy()
cv.drawContours(im_contours_filtered, contours_filtered, -1, (0, 255, 0), 2)
cv.imshow('Contours filtrés', cv.resize(im_contours_filtered, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Liste pour stocker toutes les images de pièces filtrées et leurs contours
assembled_pieces = []
all_piece_contours = []

# Zoomer et réexaminer les pièces détectées
for contour in contours_filtered:
    x, y, w, h = cv.boundingRect(contour)
    piece_image = im[y:y + h, x:x + w]
    piece_gray = cv.cvtColor(piece_image, cv.COLOR_BGR2GRAY)
    piece_before = im_contours_filtered[y:y + h, x:x + w]

    # Appliquer un flou plus fort pour réduire les détails fins du dessin
    piece_blurred = cv.GaussianBlur(piece_gray, (5, 5), 0)

    # Appliquer Sobel sur l'image floutée
    sobelx_piece = cv.Sobel(piece_blurred, cv.CV_64F, 1, 0, ksize=3)
    sobely_piece = cv.Sobel(piece_blurred, cv.CV_64F, 0, 1, ksize=3)
    sobel_piece = cv.magnitude(sobelx_piece, sobely_piece)
    sobel_piece = cv.convertScaleAbs(sobel_piece)

    # Seuillage Otsu (vous pouvez aussi essayer un seuil manuel pour plus de contrôle)
    _, th_sobel_piece = cv.threshold(sobel_piece, 50, 255, cv.THRESH_BINARY)

    # Dilatation et érosion pour renforcer les contours
    kernel = np.ones((5, 5), np.uint8)
    th_sobel_piece = cv.dilate(th_sobel_piece, kernel, iterations=1)
    th_sobel_piece = cv.erode(th_sobel_piece, kernel, iterations=1)

    # Détection des contours
    contours_piece, _ = cv.findContours(th_sobel_piece, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    im_piece_contours = piece_image.copy()
    cv.drawContours(im_piece_contours, contours_piece, -1, (0, 255, 0), 2)

    # Filtrer les contours selon leur superficie
    min_area_p = min_area  # Définir un seuil minimum pour la superficie des contours
    contours_filtered_p = [cnt_p for cnt_p in contours_piece if cv.contourArea(cnt_p) > min_area_p]

    # Filtrer les contours
    im_contours_filtered_p = piece_image.copy()
    cv.drawContours(im_contours_filtered_p, contours_filtered_p, -1, (0, 255, 0), 2)

    # Affichage des résultats
    cv.imshow(f'Contours de la pièce {x},{y}',
              np.hstack((
                  cv.resize(piece_before, WINDOW_SIZE_PIECE),  # No need to convert
                  cv.resize(cv.cvtColor(piece_gray, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
                  cv.resize(cv.cvtColor(piece_blurred, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
                  cv.resize(cv.cvtColor(sobel_piece, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
                  cv.resize(cv.cvtColor(th_sobel_piece, cv.COLOR_GRAY2BGR), WINDOW_SIZE_PIECE),
                  cv.resize(im_piece_contours, WINDOW_SIZE_PIECE),
                  cv.resize(im_contours_filtered_p, WINDOW_SIZE_PIECE)
              )))
    cv.waitKey(1000)
    cv.destroyAllWindows()

    # Ajouter l'image de la pièce avec ses contours filtrés dans la liste
    assembled_pieces.append(im_contours_filtered_p)

    # Transposer les contours trouvés aux bonnes coordonnées de l'image originale et ajouter à la liste globale des contours
    for cnt in contours_filtered_p:
        cnt += [x, y]
        all_piece_contours.append(cnt)

# Dessiner tous les contours trouvés pour chaque pièce sur l'image originale
im_all_contours = im.copy()
cv.drawContours(im_all_contours, all_piece_contours, -1, (0, 255, 0), 2)

# Afficher l'image avec tous les contours trouvés
cv.imshow('Contours filtrés', cv.resize(im_contours_filtered, WINDOW_SIZE))
cv.imshow('Tous les contours trouvés', cv.resize(im_all_contours, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()
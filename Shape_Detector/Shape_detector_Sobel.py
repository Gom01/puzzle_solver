import os
import cv2 as cv
import numpy as np

WINDOW_SIZE = (900, 700)

# Charger l'image
image_path = '../images/p1_b/Natel.Black1.jpg'
assert os.path.exists(image_path), f"Erreur : le fichier {image_path} n'existe pas."
im = cv.imread(image_path)
assert im is not None, "Erreur : l'image ne peut pas être lue."

# 6. Conversion en niveaux de gris pour la détection des contours
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cv.imshow('gray', cv.resize(gray, WINDOW_SIZE))
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
# Appliquer un seuillage automatique avec Otsu après Sobel
ret, th_sobel_otsu = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('Seuillage Sobel Otsu', cv.resize(th_sobel_otsu, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Appliquer une fermeture morphologique après Sobel (facultatif)
kernel_closing_sobel = np.ones((3, 3), np.uint8)
closing_sobel = cv.morphologyEx(sobel, cv.MORPH_CLOSE, kernel_closing_sobel)
cv.imshow('Sobel+Closing', cv.resize(closing_sobel, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

opening = cv.morphologyEx(closing_sobel, cv.MORPH_OPEN,kernel_closing_sobel)

# Détection des contours après Sobel avec Otsu
contours, _ = cv.findContours(th_sobel_otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
im_contours = im.copy()
cv.drawContours(im_contours, contours, -1, (0, 255, 0), 2)
cv.imshow('Contours détectés après Sobel Otsu', cv.resize(im_contours, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Filtrer les contours selon leur superficie
min_area = 500  # Définir un seuil minimum pour la superficie des contours
contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

# Dessiner les contours filtrés
im_contours_filtered = im.copy()
cv.drawContours(im_contours_filtered, contours_filtered, -1, (0, 255, 0), 2)
cv.imshow('Contours filtrés', cv.resize(im_contours_filtered, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Laplacian (optionnel, non utilisé mais inclus dans le commentaire)
# laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
# laplacian = cv.convertScaleAbs(laplacian)
# cv.imshow('Laplacian', cv.resize(laplacian, WINDOW_SIZE))


# Appliquer Canny (optionnel, également dans le commentaire)
# edges = cv.Canny(gray, 100, 200)
# cv.imshow('Canny', cv.resize(edges, WINDOW_SIZE))
# cv.waitKey(0)
# cv.destroyAllWindows()

# Appliquer la fermeture morphologique sur Canny (optionnel)
# closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
# cv.imshow('Canny+Closing', cv.resize(closing, WINDOW_SIZE))
# cv.waitKey(0)
# cv.destroyAllWindows()

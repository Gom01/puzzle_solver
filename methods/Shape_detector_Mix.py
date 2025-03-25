import os
import cv2 as cv
import numpy as np

WINDOW_SIZE = (900, 700)

# Charger l'image
#image_path = '../images/p1/WIN_20250306_15_09_28_Pro.jpg'
image_path = '../images/p1_b/Natel.Black1.jpg'
assert os.path.exists(image_path), f"Erreur : le fichier {image_path} n'existe pas."
im = cv.imread(image_path)
assert im is not None, "Erreur : l'image ne peut pas être lue."

# Étape 1 : Conversion en niveaux de gris
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cv.imshow('1 - Niveaux de gris', cv.resize(gray, WINDOW_SIZE))
cv.waitKey(0)

# Étape 2 : Appliquer un flou gaussien pour réduire le bruit
blurred = cv.GaussianBlur(gray, (5, 5), 0)
cv.imshow('2 - Flou gaussien', cv.resize(blurred, WINDOW_SIZE))
cv.waitKey(0)

# Étape 3 : Détection des contours avec Canny
edges = cv.Canny(blurred, 50, 150)
cv.imshow('3 - Contours Canny', cv.resize(edges, WINDOW_SIZE))
cv.waitKey(0)

# Étape 4 : Fermeture morphologique pour combler les petits trous
kernel = np.ones((3, 3), np.uint8)
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
cv.imshow('4 - Morphological Closing', cv.resize(closed, WINDOW_SIZE))
cv.waitKey(0)

# Étape 5 : Ouverture morphologique pour affiner les bords
opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
cv.imshow('5 - Morphological Opening', cv.resize(opened, WINDOW_SIZE))
cv.waitKey(0)

# Étape 6 : Détection des contours
contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
im_contours = im.copy()
cv.drawContours(im_contours, contours, -1, (0, 255, 0), 2)
cv.imshow('6 - Contours détectés', cv.resize(im_contours, WINDOW_SIZE))
cv.waitKey(0)

# Étape 7 : Approximation des contours pour les lisser
contours_approx = [cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True) for cnt in contours]
im_contours_approx = im.copy()
cv.drawContours(im_contours_approx, contours_approx, -1, (0, 255, 255), 2)
cv.imshow('7 - Contours optimisés', cv.resize(im_contours_approx, WINDOW_SIZE))
cv.waitKey(0)

cv.destroyAllWindows()

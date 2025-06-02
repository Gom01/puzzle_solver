import os
import cv2 as cv
import numpy as np

WINDOW_SIZE = (900, 700)


# Charger l'image
image_path = '../images/p1/WIN_20250306_15_09_28_Pro.jpg'
#image_path = '../images/p1_b/Natel.Black1.jpg'
assert os.path.exists(image_path), f"Erreur : le fichier {image_path} n'existe pas."
im = cv.imread(image_path)
assert im is not None, "Erreur : l'image ne peut pas être lue."

# 6. Conversion en niveaux de gris pour la détection des contours
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cv.imshow('gray', cv.resize(gray, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Seuillage pour la détection des contours
ret2, th3 = cv.threshold(gray, 160, 255, cv.THRESH_BINARY_INV)
cv.imshow('Seuillage', cv.resize(th3, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Appliquer Canny pour la détection des bords
edges = cv.Canny(gray, 30, 200)  # Seuils bas et haut

cv.imshow('Canny', cv.resize(edges, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

#https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
# Appliquer une fermeture morphologique
kernel_closing_canny = np.ones((3, 3), np.uint8)
closing_canny = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel_closing_canny)
cv.imshow('Canny+Closing', cv.resize(closing_canny, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Seuillage pour la détection des contours
ret3, th4 = cv.threshold(closing_canny, 45, 255, cv.THRESH_BINARY)
cv.imshow('Seuillage', cv.resize(th4, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Détection des contours
contours, _ = cv.findContours(th4, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
im_contours = im.copy()
cv.drawContours(im_contours, contours, -1, (0, 255, 0), 2)
cv.imshow('Contours détectés', cv.resize(im_contours, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Étape 7 : Approximation des contours pour les lisser
contours_approx = [cv.approxPolyDP(cnt, 0.001 * cv.arcLength(cnt, True), True) for cnt in contours]
im_contours_approx = im.copy()
cv.drawContours(im_contours_approx, contours_approx, -1, (0, 255, 255), 2)
cv.imshow('7 - Contours optimisés', cv.resize(im_contours_approx, WINDOW_SIZE))
cv.waitKey(0)

cv.destroyAllWindows()


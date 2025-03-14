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

# Appliquer le filtre Laplacien pour les gradients
laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
laplacian = cv.convertScaleAbs(laplacian)

cv.imshow('Laplacian', cv.resize(laplacian, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

#https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
# Appliquer une fermeture morphologique
kernel_closing_laplacian = np.ones((3, 3), np.uint8)
closing_laplacian = cv.morphologyEx(laplacian, cv.MORPH_CLOSE, kernel_closing_laplacian)
cv.imshow('Laplacian+Closing', cv.resize(closing_laplacian, WINDOW_SIZE))
cv.waitKey(0)
cv.destroyAllWindows()

# Seuillage pour la détection des contours
ret3, th4 = cv.threshold(closing_laplacian, 45, 255, cv.THRESH_BINARY)
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

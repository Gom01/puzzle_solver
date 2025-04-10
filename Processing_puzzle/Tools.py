import cv2 as cv
import numpy as np

#Basic settings
WINDOW_SIZE = (1000, 800)
WINDOW_SIZE_PIECE = (200, 200)
X_PERCENT = 50  # Ajustable (ex : 30% autour de la médiane)
MARGIN_PERCENT = 1  # Ajuste la taille de la marge autour des contours

class Tools:
    """
    Cette classe contient des méthodes utilitaires pour le traitement d'image avec OpenCV.
    Chaque méthode peut afficher l'image résultante selon les besoins.
    """
    def convert_to_grayscale(self, image, window=False, window_size=None, time=0):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Grayscale', cv.resize(gray, window_size))
            cv.waitKey(0)
        return gray

    def apply_otsu_threshold(self, image,window=False, window_size=None, time=0):
        blur = cv.GaussianBlur(image, (5,5), 0)
        ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Otsu Threshold', cv.resize(th3, window_size))
            cv.waitKey(0)
        return th3

    def remove_noise(self, image, kernel_size,kernel_size2, iterations, window=False, window_size=None, time=0):
        kernel = np.ones(kernel_size, np.uint8)
        morpho = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
        kernel = np.ones(kernel_size2, np.uint8)
        closing = cv.morphologyEx(morpho, cv.MORPH_CLOSE, kernel)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Noise Removal', cv.resize(closing, window_size))
            cv.waitKey(0)
        return closing

    def find_contours(self,image_origin, image, window=False, window_size=None, time=0):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if window:
            image_c = cv.drawContours(image_origin.copy(), contours, -1, (0, 255, 0), 2)
            cv.imshow('Contours', cv.resize(image_c, window_size))
            cv.waitKey(time)
        return contours,_

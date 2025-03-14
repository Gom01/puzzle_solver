import cv2 as cv
import numpy as np

class Piece_Puzzle:
    def __init__(self, image, contours, x, y):
        """
        Initialise une pièce de puzzle.

        :param image: L'image de la pièce avec le fond noir et l'intérieur blanc.
        :param contours: Les contours filtrés de la pièce.
        :param x: Coordonnée x de la pièce.
        :param y: Coordonnée y de la pièce.
        """
        self.image = image
        self.contours = contours
        self.x = x
        self.y = y

    def get_image(self):
        """Retourne l'image de la pièce."""
        return self.image

    def get_contours(self):
        """Retourne les contours de la pièce."""
        return self.contours

    def get_position(self):
        """Retourne la position (x, y) de la pièce."""
        return self.x, self.y

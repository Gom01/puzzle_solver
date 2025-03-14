import cv2 as cv
import numpy as np

class Piece_Puzzle:
    def __init__(self, image,image_color, contours, x, y):
        """
        Initialise une pièce de puzzle.

        :param image: L'image de la pièce avec le fond noir et l'intérieur blanc.
        :param contours: Les contours filtrés de la pièce.
        :param x: Coordonnée x de la pièce.
        :param y: Coordonnée y de la pièce.
        """
        self.image = image
        self.image_color = image_color
        self.contours = contours
        self.x = x
        self.y = y

    def get_image(self):
        """Retourne l'image de la pièce."""
        return self.image

    def get_image_color(self):
        """Retourne l'image de la pièce."""
        return self.image_color

    def get_contours(self):
        adjusted_contours = []

        for contour in self.contours:
            adjusted_contour = contour - np.array([[self.x, self.y]])
            adjusted_contours.append(adjusted_contour)


        return adjusted_contours


    def get_position(self):
            """Retourne la position (x, y) de la pièce."""
            return self.x, self.y

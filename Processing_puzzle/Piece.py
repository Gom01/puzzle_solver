import cv2 as cv
import numpy as np
from django.utils.timezone import override



class Piece:
    def __init__(self, image_black_white,image_color, contours, x, y, index):
        """
        Initialise une pièce de puzzle.
        :param image_black_white: L'image de la pièce avec le fond noir et l'intérieur blanc.
        :param contours: Les contours filtrés de la pièce.
        :param x: Coordonnée x de la pièce.
        :param y: Coordonnée y de la pièce.
        """
        self.image_black_white = image_black_white
        self.image_color = image_color
        self.contours = contours
        self.x = x
        self.y = y
        self.index = index

        self.adjusted_contours = None
        self.colors_contour = []
        self.corners = None
        self.side_1 = None
        self.side_2 = None
        self.side_3 = None
        self.side_4 = None

        self.side_1_info = None
        self.side_2_info = None
        self.side_3_info = None
        self.side_4_info = None

        self.side_1_eq = None
        self.side_2_eq = None
        self.side_3_eq = None
        self.side_4_eq = None

        self.name_piece = ""

        self.side_1_name = ""
        self.side_2_name = ""
        self.side_3_name = ""
        self.side_4_name = ""

        self.can_be_process = True


    def get_black_white_image(self):
        return self.image_black_white

    def get_color_image(self):
        return self.image_color

    def get_corners(self):
        return self.corners

    def get_position(self):
        return self.x, self.y

    def get_contours(self):
        return self.contours

    def get_color_contour(self):
        return self.colors_contour

    def set_corners(self, corners):
        self.corners = corners
        return

    def set_sides(self, side1, side2, side3, side4):
        self.side_1, self.side_2, self.side_3, self.side_4 = side1, side2, side3, side4
        return

    def get_4_sides(self):
        return self.side_1, self.side_2, self.side_3, self.side_4

    def set_4_sides_info(self,side1, side2, side3, side4):
        self.side_1_info, self.side_2_info, self.side_3_info, self.side_4_info = side1, side2, side3, side4
        return

    def get_4_sides_info(self):
        return self.side_1_info, self.side_2_info, self.side_3_info, self.side_4_info

    def set_4_sides_eq(self, side1, side2, side3, side4):
        self.side_1_eq, self.side_2_eq, self.side_3_eq, self.side_4_eq = side1, side2, side3, side4
        return

    def get_4_sides_eq(self):
        return self.side_1_eq, self.side_2_eq, self.side_3_eq, self.side_4_eq

    def set_name(self,name):
        self.name_piece = name
        return
    def get_name_piece(self):
        return self.name_piece

    def set_4_sides_name(self,side1, side2, side3, side4):
        self.side_1_name, self.side_2_name, self.side_3_name, self.side_4_name = side1, side2, side3, side4
        return

    def get_4_sides_name(self):
        return self.side_1_name, self.side_2_name, self.side_3_name, self.side_4_name



    def __str__(self):
        return (f"Piece: {self.index}\n  Position: ({self.x},{self.y})\n  Contours: [{self.contours[0]},...]\n  "
                f"Corners: [{self.corners[0]},...]\n  Side1:[{self.side_1[0]}]\n  Colored Contours : [{self.colors_contour[0]},...]\n  ")



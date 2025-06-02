import cv2
import numpy as np
class Piece:
    def __init__(self, image_black_white,image_color, contours, index):
        """
        Initialise une pièce de puzzle.
        :param image_black_white: L'image de la pièce avec le fond noir et l'intérieur blanc.
        :param contours: Les contours filtrés de la pièce.
        """
        self.image_black_white = image_black_white
        self.image_straigthen = None
        self.image_color = image_color
        self.contours = contours
        self.index = index


        self.x = None
        self.y = None
        self.colors_contour = []
        self.corners = None


        self.sides = []

        self.moment = None

        self.name_piece = ""
        self.number_rotation = 0
        self.is_bad = False


    def set_sides(self, side1, side2, side3, side4):
        self.sides = side1, side2, side3, side4
        return

    def get_strait_image(self):
        return self.image_straigthen

    def set_strait_image(self, image):
        self.image_straigthen = image
        return


    def get_sides(self):
        return self.sides # [left, bottom, right, top]

    def get_sides_info(self):
        sides = self.sides
        return sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info()


    def get_number_rotation(self):
        return self.number_rotation

    def reset_piece(self, piece):
        self.image_black_white = piece.get_black_white_image()
        self.image_color = piece.get_color_image()
        self.image_straigthen = piece.get_strait_image()
        self.contours = piece.get_contours()
        self.index = piece.get_index()
        self.x = piece.get_position()[0]
        self.y = piece.get_position()[1]
        self.colors_contour = piece.get_color_contour()
        self.corners = piece.get_corners()
        self.sides = piece.get_sides()
        self.moment = piece.get_moment()
        self.name_piece = piece.get_name()
        self.number_rotation = piece.get_number_rotation()
        return

    def increment_number_rotation(self):
        self.number_rotation = self.number_rotation + 1
        return

    def set_number_rotation(self, i):
        self.number_rotation = i
        return

    def get_black_white_image(self):
        return self.image_black_white

    def get_color_image(self):
        return self.image_color

    def set_color_image(self, newImg):
        self.image_color = newImg
        return

    def get_index(self):
        return self.index

    def index_to_piece(self, index):
        return self

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

    def set_name(self,name):
        self.name_piece = name
        return

    def get_name(self):
        return self.name_piece

    def set_moment(self, moment):
        self.moment = moment
        return

    def set_position(self, position):
        self.x = position[0]
        self.y = position[1]
        return

    def get_moment(self):
        return self.moment


    def __str__(self):
        s1,s2,s3,s4 = self.get_sides_info() # [left, top, right, down]
        return  f"(P:{self.index}: |{s1}|{s2}|{s3}|{s4}|)"


    def rotate_image_by_rotation(self):
        img = self.image_color

        k = int(self.number_rotation) % 4

        if k == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif k == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif k == 3:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # If k == 0, no rotation needed

        return img

    def rotate_piece_by_corners(self):
        """
        Cette méthode ajuste l'orientation de la pièce en fonction de ses coins.
        Elle aligne le côté 1 (le côté bas de la pièce) avec l'axe X.
        """
        # Supposons que les coins sont dans `self.corners` et qu'ils sont dans un ordre précis
        # Par exemple, le coin bas-gauche et le coin bas-droit correspondent à `side1`.
        sorted_corners = sorted(self.corners, key=lambda x: x[1], reverse=True)
        bottom_left_corner = sorted_corners[0]
        bottom_right_corner = sorted_corners[1]

        print("Coins bas (pour aligner le côté 1) :", bottom_left_corner, bottom_right_corner)

        x1, y1 = bottom_left_corner
        x2, y2 = bottom_right_corner

        # Calculer l'angle pour aligner les coins bas avec l'axe X
        dx = x2 - x1
        dy = y2 - y1

        # Calculer l'angle entre les coins et l'axe X
        angle = np.arctan2(dy, dx)
        angle_degrees = np.degrees(angle)  # Convertir en degrés
        if abs(angle_degrees) < 10 or abs(abs(angle_degrees) - 180) < 10:
            angle_degrees = 0

        print(f"Angle pour aligner les coins avec l'axe X : {angle_degrees}°")

        # Appliquer la rotation pour aligner les coins bas avec l'axe X
        self.image_color = self.rotate_image_by_angle(angle_degrees)

        # Appliquer la rotation en fonction de `self.number_rotation`
        return self.image_color

    def rotate_image_by_angle(self, angle_degrees):
        """
        Effectue la rotation de l'image en fonction de l'angle donné.
        """
        rows, cols = self.image_color.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle_degrees, 1)

        # Appliquer la rotation
        rotated_image = cv2.warpAffine(self.image_color, rotation_matrix, (cols, rows))
        return rotated_image
import cv2
class Piece:
    def __init__(self, image_black_white,image_color, contours, index):
        """
        Initialise une pièce de puzzle.
        :param image_black_white: L'image de la pièce avec le fond noir et l'intérieur blanc.
        :param contours: Les contours filtrés de la pièce.
        """
        self.image_black_white = image_black_white
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

    def get_sides(self):
        return self.sides

    def get_sides_info(self):
        sides = self.sides
        return sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info()


    def get_number_rotation(self):
        return self.number_rotation

    def reset_piece(self, piece):
        self.image_black_white = piece.get_black_white_image()
        self.image_color = piece.get_color_image()
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

    def get_moment(self):
        return self.moment


    def __str__(self):
        s1,s2,s3,s4 = self.get_sides_info() # [left, top, right, down]
        return  f"(P:{self.index}: |{s1}|{s2}|{s3}|{s4}|)"


    def rotate(self):
        img = self.image_color
        number_rotation = self.number_rotation

        # Normalize the rotation count to 0–3
        k = int(number_rotation) % 4

        if k == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif k == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif k == 3:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # If k == 0, no rotation needed
        self.image_color = img
        self.number_rotation = 0


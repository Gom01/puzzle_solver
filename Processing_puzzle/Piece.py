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
        self.adjusted_contours = None
        self.colors_contour = []
        self.corners = None


        self.sides = []
        self.sides_color = []

        self.moment = None


        self.name_piece = ""

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
        self.sides = [side1,side2,side3,side4]
        return

    def set_sides_color(self, side1, side2, side3, side4):
        self.sides_color = [side1,side2,side3,side4]
        return

    def get_sides(self):
        return self.sides


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
        return (f"Piece: {self.index}\n  Position: ({self.x},{self.y})\n  Contours: [{self.contours[0]},...]\n  "
                f"Corners: [{self.corners[0]},...]\n  Side1:[{self.sides[0]}]\n  Colored Contours : [{self.colors_contour[0]},...]\n  ")



class Side:
    def __init__(self):
        self.side_name = ""
        self.side_info = None
        self.side_points = None
        self.side_colors = None
        self.side_eq = None
        self.side_hierarchy = None
        self.side_match = None
        self.piece_match = None
        self.rotation = None
        self.number_of_rotation = 0



    def set_side_name(self, side_name):
        self.side_name = side_name
        return

    def set_side_info(self, side_info):
        self.side_info = side_info
        return

    def set_side_points(self, side_points):
        self.side_points = side_points
        return

    def set_side_colors(self, side_colors):
        self.side_colors = side_colors
        return

    def set_side_eq(self, side_eq):
        self.side_eq = side_eq
        return

    def set_side_hierarchy(self, side_hierarchy):
        self.side_hierarchy = side_hierarchy
        return

    def get_side_hiearchy(self):
        return self.side_hierarchy


    def get_side_name(self):
        return self.side_name

    def get_side_info(self):
        return self.side_info

    def get_side_points(self):
        return self.side_points

    def get_side_colors(self):
        return self.side_colors

    def get_side_eq(self):
        return self.side_eq

    def set_piece_match(self, piece_match):
        if piece_match is None:
            self.piece_match = piece_match
        return

    def set_side_match(self, side_match):
        if side_match is None:
            self.side_match = side_match
        return

    def get_piece_match(self):
        return self.piece_match

    def get_side_match(self):
        return self.side_match

    def set_rotation(self, rotation):
        self.rotation = rotation
        return
    def get_rotation(self):
        return self.rotation


    def __str__(self):
        return (f"Side : {self.side_name}\n, points : {self.side_points}\n, colors : {self.side_colors}\n, equation : {self.side_eq}, hierarchy : {self.side_hierarchy}")
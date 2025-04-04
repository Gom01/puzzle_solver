class Side:
    def __init__(self):
        self.side_name = ""
        self.side_info = None
        self.side_points = None
        self.side_colors = None
        self.side_eq = None



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
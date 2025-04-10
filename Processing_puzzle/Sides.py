class Side:
    def __init__(self, side_contour):
        self.side_info = None #[-1,1,1,0]
        self.side_contour = side_contour #[...,...,...]
        self.side_color = None #[[...,..,...],[...,...,..],..]

    def set_side_info(self, side_info):
        self.side_info = side_info
        return

    def get_side_info(self):
        return self.side_info

    def set_side_color(self, side_color):
        self.side_color = side_color
        return

    def get_side_color(self):
        return self.side_color

    def set_side_contour(self, side_contour):
        self.side_contour = side_contour
        return

    def get_side_contour(self):
        return self.side_contour



    def __str__(self):
        return f"{self.side_info}"
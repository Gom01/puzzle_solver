import numpy as np
class Side:
    def __init__(self, side_contour):
        self.side_info = None #[-1,1,1,0]
        self.side_contour = side_contour #[...,...,...]
        self.side_color = None #[[...,..,...],[...,...,..],..]
        self.side_size = None

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


    def get_side_size(self):
        if self.side_contour == 2:
            return 2
        if self.side_size is None :
            contour = self.side_contour
            start_point = np.array(contour[0])
            end_point = np.array(contour[-1])
            distance = np.linalg.norm(end_point - start_point)
            self.side_size = distance

        return self.side_size

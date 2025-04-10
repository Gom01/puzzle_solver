import cv2
import numpy as np

import numpy as np
import cv2

def straighten_piece(piece):
    center = piece.get_position()
    corners = piece.get_corners()
    img = piece.get_color_image()

    # Function to check if a line is close to horizontal
    def is_horizontal(pt1, pt2, threshold=5):
        return abs(pt1[1] - pt2[1]) <= threshold

    # Check if corners[0] and corners[1] form a horizontal line
    pt1, pt2 = corners[0], corners[1]
    if not is_horizontal(pt1, pt2):
        # If corners[0] and corners[1] are not horizontal, check corners[1] and corners[2]
        pt1, pt2 = corners[1], corners[2]
        if not is_horizontal(pt1, pt2):
            # If corners[1] and corners[2] are also not horizontal, use corners[2] and corners[3]
            pt1, pt2 = corners[2], corners[3]
            if not is_horizontal(pt1, pt2):
                print("No horizontal edges found. Exiting.")
                return  # No suitable horizontal edge found, exit.

    # Calculate the angle of the line
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Only rotate if the angle is above a certain threshold
    if abs(angle) > 1:  # Adjust this threshold as needed
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        (h, w) = img.shape[:2]
        rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))

        cv2.imshow('rotated_image', rotated_image)
        cv2.waitKey(0)

        piece.set_color_image(rotated_image)
    else:
        # If the angle is small, don't apply any rotation
        print("Piece is already aligned.")

    return




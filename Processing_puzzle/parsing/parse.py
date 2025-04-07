import cv2 as cv
import scipy
import numpy as np

from Processing_puzzle.Piece import Piece
from Processing_puzzle.Tools import Tools

'''
This function takes the image (taken with a phone) and a puzzle object and then create pickle file containing
basics information on the pieces :  
- Black and White image
- Colored image
- Contours
'''

def parse_image(image_path, puzzle):

    #Basics tools and settings (size image and margin arround)
    tools = Tools()
    new_width = 500
    margin_scale = 1.10
    scale_factor = 5.0

    #Modifying the size of the image (so I can see it on my screen)
    image = cv.imread(image_path)
    (h, w) = image.shape[:2]
    scale = new_width / w
    new_height = int(h * scale)
    image = cv.resize(image, (new_width, new_height))


    # Transformations to find contours of the 24 pieces
    im_gray = tools.convert_to_grayscale(image, window=False, window_size=(new_width, new_height), time=0)
    im_thresh = tools.apply_otsu_threshold(im_gray, window=False, window_size=(new_width, new_height), time=0)
    im_clean = tools.remove_noise(im_thresh, (5, 5), (2, 2), iterations=20, window=False, window_size=(new_width, new_height),time=0)
    contours, _ = tools.find_contours(image, im_clean, window=False, window_size=(new_width, new_height), time=0)
    print(f"Number of pieces found : {len(contours)}")

    #Go through all the pieces
    for i, cnt in enumerate(contours):

        #Detouring the pieces from the main image:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int64(box)
        box_float = np.float32(box)
        center = np.mean(box_float, axis=0)
        expanded_box = (box_float - center) * margin_scale + center
        expanded_box = np.float32(expanded_box)

        #To see what's going on the base image
        #cv.drawContours(image, [np.int32(expanded_box)], 0, (0, 255, 0), 2)


        #Create colored image (with margins)
        width = int(rect[1][0] * margin_scale)
        height = int(rect[1][1] * margin_scale)
        dst_pts = np.array([[0, height - 1],[0, 0],[width - 1, 0],[width - 1, height - 1]], dtype="float32")
        M = cv.getPerspectiveTransform(expanded_box, dst_pts)
        warped = cv.warpPerspective(image, M, (width, height))
        (h, w) = warped.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        piece_image = cv.resize(warped, new_size, interpolation=cv.INTER_CUBIC)
        piece_image_2 = piece_image.copy()  # For green dots
        piece_image_3 = piece_image.copy()  # For masked color

        # Transformation on each piece (find contour)
        gray_piece = tools.convert_to_grayscale(piece_image, window=False, window_size=new_size, time=0)
        thresh_piece = tools.apply_otsu_threshold(gray_piece, window=False, window_size=new_size, time=0)
        cleaned_piece = tools.remove_noise(thresh_piece, (4, 4), (15, 15), iterations=20, window=False,window_size=new_size, time=0)
        contoured_piece, _ = tools.find_contours(piece_image, cleaned_piece, window=False, window_size=new_size, time=0)

        # Keep only the important contour (which contains maximum of points)
        contoured_piece = max(contoured_piece, key=cv.contourArea)

        # Smooth contour and convert it to points
        contoured_piece = convert_contour(contoured_piece)
        smoothed = gaussian_smooth(contoured_piece)

        # Create mask for black and white and colored image
        contour_np = np.array(smoothed, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(piece_image_3.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [contour_np], -1, 255, cv.FILLED)

        # Apply mask to color image
        masked_image = cv.bitwise_and(piece_image_2, piece_image_2, mask=mask)

        # Black and white binary image
        binary_image = np.zeros_like(mask)
        cv.drawContours(binary_image, [contour_np], -1, 255, cv.FILLED)

        # Show results
        #cv2.imshow("Masked Color", masked_image)
        #cv2.imshow("Black & White Mask", binary_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        puzzle.add_piece(Piece(binary_image, masked_image, smoothed, i))
    puzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Puzzle parsed...")
    return


def gaussian_smooth(points, sigma=5):
    points_np = np.array(points, dtype=np.float32)
    x, y = points_np[:, 0], points_np[:, 1]

    x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma)
    y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma)

    return list(zip(x_smooth.astype(int), y_smooth.astype(int)))

def convert_contour(contours):
    contour_points = [tuple(pt[0]) for pt in contours]
    return contour_points

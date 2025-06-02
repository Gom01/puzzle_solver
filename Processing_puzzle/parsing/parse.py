import cv2 as cv
import numpy as np
import scipy
from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from Processing_puzzle.Piece import Piece
from Processing_puzzle.Tools import Tools


def convert_contour(contours):
    return [tuple(pt[0]) for pt in contours]

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def show_subtracted_mask_with_biggest_contour(image_paths, threshold=120, debug_resize_width=800):
    # Load images
    images = [cv.imread(p) for p in image_paths]
    base = images[0]
    diff_accum = np.zeros_like(base, dtype=np.uint8)

    # Subtract all images from the base
    for i in range(1, len(images)):
        diff = cv.absdiff(base, images[i])
        diff_accum = cv.bitwise_or(diff_accum, diff)

    # Convert to grayscale and threshold to create a mask
    diff_gray = cv.cvtColor(diff_accum, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(diff_gray, threshold, 255, cv.THRESH_BINARY_INV)

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Overlay only the biggest contour
    mask_with_contour = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    if contours:
        biggest = max(contours, key=cv.contourArea)
        cv.drawContours(mask_with_contour, [biggest], -1, (0, 255, 0), 2)
    else:
        print("⚠️ No contour found.")

    # Resize for better display
    scale = debug_resize_width / mask_with_contour.shape[1]
    resized = cv.resize(mask_with_contour, (debug_resize_width, int(mask_with_contour.shape[0] * scale)))

    # Show the image
    plt.figure(figsize=(10, 8))
    plt.imshow(resized)
    plt.title("Subtracted Binary Mask with Biggest Contour")
    plt.axis("off")
    plt.show()






def gaussian_smooth(points, sigma=5):
    points_np = np.array(points, dtype=np.float32)
    x, y = points_np[:, 0], points_np[:, 1]
    x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma)
    y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma)
    return list(zip(x_smooth.astype(int), y_smooth.astype(int)))

def get_foreground_mask(images, threshold=30):
    gray_images = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in images]
    base = gray_images[0]
    diff_sum = np.zeros_like(base)
    for img in gray_images[1:]:
        diff = cv.absdiff(base, img)
        diff_sum = cv.bitwise_or(diff_sum, diff)
    _, mask = cv.threshold(diff_sum, threshold, 255, cv.THRESH_BINARY_INV)
    return mask

def parse_image_multi(image_paths, puzzle, window=False):
    tools = Tools()
    new_width = 500
    margin_scale = 1.8
    scale_factor = 5.0

    # Load and resize all images
    first_img = cv.imread(image_paths[0])
    show_subtracted_mask_with_biggest_contour(image_paths)
    x, y, w, h = None

    # Apply same crop to all images
    images = []
    for path in image_paths:
        img = cv.imread(path)
        cropped = img[y:y + h, x:x + w]
        scale = new_width / cropped.shape[1]
        new_height = int(cropped.shape[0] * scale)
        resized = cv.resize(cropped, (new_width, new_height))
        images.append(resized)

    # Get foreground mask
    mask = get_foreground_mask(images)

    if window:
        cv.imshow("Foreground Mask", mask)
        cv.waitKey(0)
        cv.destroyWindow("Foreground Mask")

    # Use the first image as base for color information
    base_image = images[0]

    # Clean and find contours
    im_clean = tools.remove_noise(mask, (5, 5), (2, 2), iterations=15, window=False, window_size=(new_width, new_height), time=0)
    contours, _ = tools.find_contours(base_image, im_clean, window=False, window_size=(new_width, new_height), time=0)
    print(f"Number of pieces found: {len(contours)}")

    if window:
        debug_img = base_image.copy()
        cv.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        cv.imshow("Detected Contours", debug_img)
        cv.waitKey(0)
        cv.destroyWindow("Detected Contours")

    # Process each piece
    for i, cnt in enumerate(contours):
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int64(box)
        box_float = np.float32(box)
        center = np.mean(box_float, axis=0)
        expanded_box = (box_float - center) * margin_scale + center
        expanded_box = np.float32(expanded_box)

        width = int(rect[1][0] * margin_scale)
        height = int(rect[1][1] * margin_scale)
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
        M = cv.getPerspectiveTransform(expanded_box, dst_pts)
        warped = cv.warpPerspective(base_image, M, (width, height))
        (h, w) = warped.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        piece_image = cv.resize(warped, new_size, interpolation=cv.INTER_CUBIC)
        piece_image_2 = piece_image.copy()
        piece_image_3 = piece_image.copy()

        gray_piece = tools.convert_to_grayscale(piece_image, window=False, window_size=new_size, time=0)
        thresh_piece = tools.apply_otsu_threshold(gray_piece, window=False, window_size=new_size, time=0)
        cleaned_piece = tools.remove_noise(thresh_piece, (4, 4), (15, 15), iterations=20, window=False, window_size=new_size, time=0)
        contoured_piece, _ = tools.find_contours(piece_image, cleaned_piece, window=False, window_size=new_size, time=0)

        contoured_piece = max(contoured_piece, key=cv.contourArea)
        contoured_piece = convert_contour(contoured_piece)
        smoothed = gaussian_smooth(contoured_piece)

        contour_np = np.array(smoothed, dtype=np.int32).reshape((-1, 1, 2))
        mask_piece = np.zeros(piece_image_3.shape[:2], dtype=np.uint8)
        cv.drawContours(mask_piece, [contour_np], -1, 255, cv.FILLED)

        masked_image = cv.bitwise_and(piece_image_2, piece_image_2, mask=mask_piece)
        binary_image = np.zeros_like(mask_piece)
        cv.drawContours(binary_image, [contour_np], -1, 255, cv.FILLED)

        puzzle.add_piece(Piece(binary_image, masked_image, smoothed, i))

    puzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Puzzle parsed with multi-background approach.")

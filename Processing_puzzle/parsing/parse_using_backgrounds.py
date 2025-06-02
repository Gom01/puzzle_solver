import cv2
import numpy as np

from Processing_puzzle.Piece import Piece
from Processing_puzzle.Puzzle import Puzzle

def convert_contour(contours):
    return [tuple(pt[0]) for pt in contours]

def process_puzzle_images(image_paths, puzzle, window=False):
    # Load and downscale images
    color_images = [cv2.resize(cv2.imread(path), (0, 0), fx=0.5, fy=0.5) for path in image_paths]

    # Step 1: Compute color difference
    diff = np.zeros_like(color_images[0])
    for i in range(len(color_images)):
        for j in range(i + 1, len(color_images)):
            diff |= cv2.absdiff(color_images[i], color_images[j])
    if window:
        cv2.imshow("Step 1 - Color Difference", cv2.resize(diff, (0, 0), fx=0.5, fy=0.5))

    # Step 2: Threshold to detect screen
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_diff, (9, 9), 0)
    _, binary_mask = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if window:
        cv2.imshow("Step 2 - Initial Binary Mask (Screen Detection)", cv2.resize(binary_mask, (0, 0), fx=0.5, fy=0.5))

    # Step 3: Find largest rectangular contour (screen)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour, max_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and area > max_area:
            largest_contour = approx
            max_area = area

    # Step 4: Crop all images to the screen region
    cropped_images = []
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        for img in color_images:
            cropped_images.append(img[y:y+h, x:x+w])
        if window:
            cv2.imshow("Step 3 - Cropped Region (Screen Only)", cv2.resize(cropped_images[0], (0, 0), fx=0.5, fy=0.5))
    else:
        print("No valid rectangular contour found.")
        return

    # Step 5: Compute difference again from cropped images
    cropped_diff = np.zeros_like(cropped_images[0])
    for i in range(len(cropped_images)):
        for j in range(i + 1, len(cropped_images)):
            cropped_diff |= cv2.absdiff(cropped_images[i], cropped_images[j])

    gray_cropped_diff = cv2.cvtColor(cropped_diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_cropped_diff, (15, 15), 0)
    _, final_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if window:
        cv2.imshow("Step 4 - Binary Mask of Differences (Puzzle)", cv2.resize(final_mask, (0, 0), fx=0.5, fy=0.5))

    # Step 6: Find detailed contours of puzzle pieces
    all_contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in all_contours]
    if not areas:
        print("No contours found.")
        return

    min_area = np.mean(areas) / 10.0
    max_area = np.mean(areas) * 10
    filtered_contours = [c for c in all_contours if min_area < cv2.contourArea(c) < max_area]

    if window:
        contour_display = cropped_images[0].copy()
        cv2.drawContours(contour_display, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Step 5 - Filtered Contours Drawn", cv2.resize(contour_display, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(0)

    # Step 7: Extract and isolate each puzzle piece
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        pad = 50
        x_start = max(x - pad, 0)
        y_start = max(y - pad, 0)
        x_end = min(x + w + pad, cropped_images[0].shape[1])
        y_end = min(y + h + pad, cropped_images[0].shape[0])

        piece = cropped_images[0][y_start:y_end, x_start:x_end].copy()
        shifted_contour = contour - [x_start, y_start]

        # 1. Create empty mask
        mask = np.zeros(piece.shape[:2], dtype=np.uint8)

        # 2. Draw the contour onto the mask
        cv2.drawContours(mask, [shifted_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # 3. Erode the drawn mask (now that it's filled)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.erode(mask, kernel, iterations=3)

        # 4. Apply the mask to isolate the piece from black background
        isolated = cv2.bitwise_and(piece, piece, mask=mask_eroded)

        # Enhance only non-black pixels
        alpha = 1.2  # contrast
        beta = 30  # brightness

        # Create a mask where pixels are not black
        non_black_mask = np.any(isolated > 0, axis=-1)

        # Extract non-black pixels
        enhanced = isolated.copy()
        non_black_pixels = isolated[non_black_mask]

        # Apply contrast + brightness to non-black pixels
        enhanced_pixels = cv2.convertScaleAbs(non_black_pixels, alpha=alpha, beta=beta)
        enhanced[non_black_mask] = enhanced_pixels

        # Replace isolated image with enhanced version
        isolated = enhanced

        if window:
            cv2.imshow(f"Piece {i + 1} - Binary Mask", mask)
            cv2.imshow(f"Piece {i + 1} - Isolated on Black", isolated)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Piece {i + 1} - Binary Mask")
            cv2.destroyWindow(f"Piece {i + 1} - Isolated on Black")

        puzzle.add_piece(Piece(mask_eroded, isolated, convert_contour(shifted_contour), i))

    puzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("🧩 Puzzle parsed with multi-background approach.")
    print(f"🧩 Number of pieces found : {len(filtered_contours)}")
from sklearn.cluster import KMeans
import numpy as np
import cv2
from Processing_puzzle import Puzzle as p



def find_color(puzzle, window=False):

    pieces = puzzle.get_pieces()

    def get_multiple_dominant_colors(pixels, k=3):
        """Returns the k dominant colors using K-means clustering."""
        pixels = np.array(pixels)
        kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_.astype(int)

    for idx, piece in enumerate(pieces):
        contours = piece.get_contours()

        M = cv2.moments(np.array(contours))
        cx,cy = 0,0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        piece.x, piece.y = cx, cy
        img = piece.get_color_image()
        sides_img = img.copy()
        center = piece.get_position()
        cx, cy = center

        if window:
            cv2.circle(sides_img, center, 5, (255, 0, 0), -1)

        # Step 1: Detect intersection points on vertical/horizontal axis
        top = bottom = left = right = None
        t = 3  # axis tolerance
        circle_radius = 50  # radius for collecting pixels
        for pt in contours:
            x, y = pt
            if cx - t <= x <= cx + t and y < cy:
                top = pt
            elif cx - t <= x <= cx + t and y > cy:
                bottom = pt
            elif cy - t <= y <= cy + t and x < cx:
                left = pt
            elif cy - t <= y <= cy + t and x > cx:
                right = pt

        axis_points = [top, bottom, left, right]
        if window:
            for pt in axis_points:
                cv2.circle(sides_img, pt, circle_radius, (0, 0, 255), 1)

        # Step 2: Mask only the puzzle piece area
        mask = np.zeros(sides_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [np.array(contours)], -1, 255, -1)

        # Step 3: Sample pixels in each side circle
        top_side, bottom_side, left_side, right_side = [], [], [], []
        height, width = mask.shape
        sampling_stride = 3  # Controls how dense pixel sampling is
        MAX_SAMPLES = 1000  # Limit max pixels per side

        for y in range(0, height, sampling_stride):
            for x in range(0, width, sampling_stride):
                if mask[y, x] == 0:
                    continue  # skip outside pixels

                pt = np.array([x, y])
                color = img[y, x]

                if top is not None and np.linalg.norm(pt - np.array(top)) <= circle_radius:
                    top_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (0, 0, 255), -1)  # Red
                elif bottom is not None and np.linalg.norm(pt - np.array(bottom)) <= circle_radius:
                    bottom_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (0, 255, 0), -1)  # Green
                elif left is not None and np.linalg.norm(pt - np.array(left)) <= circle_radius:
                    left_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (255, 0, 0), -1)  # Blue
                elif right is not None and np.linalg.norm(pt - np.array(right)) <= circle_radius:
                    right_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (0, 255, 255), -1)  # Yellow

        # Step 4: Clamp to max samples
        top_side = top_side[:MAX_SAMPLES]
        bottom_side = bottom_side[:MAX_SAMPLES]
        left_side = left_side[:MAX_SAMPLES]
        right_side = right_side[:MAX_SAMPLES]

        # Step 5: Get dominant colors
        top_colors = get_multiple_dominant_colors(top_side)
        bottom_colors = get_multiple_dominant_colors(bottom_side)
        left_colors = get_multiple_dominant_colors(left_side)
        right_colors = get_multiple_dominant_colors(right_side)

        # print(f"\n--- Piece {idx} ---")
        # print(f"Top colors: {top_colors}")
        # print(f"Bottom colors: {bottom_colors}")
        # print(f"Left colors: {left_colors}")
        # print(f"Right colors: {right_colors}")

        # Step 6: Display dominant color bars on the image
        def ensure_bgr(color):
            return (int(color[0]), int(color[1]), int(color[2]))

        x_offset = 0
        y_offset = 0
        color_width = 20
        color_height = 40

        def draw_colors(colors, x_off):
            for i, color in enumerate(colors):
                bgr = ensure_bgr(color)
                y = y_offset + i * (color_height + 5)
                if window:
                    cv2.rectangle(sides_img, (x_off, y), (x_off + color_width, y + color_height), bgr, -1)

        draw_colors(top_colors, x_offset)
        draw_colors(bottom_colors, x_offset + 60)
        draw_colors(left_colors, x_offset + 120)
        draw_colors(right_colors, x_offset + 180)

        # Step 7: Show result
        if window:
            cv2.imshow(f"Piece {idx} - Dominant Colors", sides_img)
            cv2.waitKey(0)

        s1,s2,s3,s4 = piece.get_sides() # [left, bottom, right, top]
        s1.set_side_color(left_colors)
        s2.set_side_color(bottom_colors)
        s3.set_side_color(right_colors)
        s4.set_side_color(top_colors)

    if window:
        cv2.destroyAllWindows()

    print("Color found..")
    return

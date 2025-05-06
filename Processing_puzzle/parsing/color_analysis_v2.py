from sklearn.cluster import KMeans
import numpy as np
import cv2
from Processing_puzzle import Puzzle as p


def find_color(puzzle, window=False, k_dominant_colors=5):
    pieces = puzzle.get_pieces()

    def get_multiple_dominant_colors(pixels, k=5):
        """Returns k dominant colors and their weights (pixel proportion)."""
        pixels = np.array(pixels)
        if len(pixels) == 0:
            return np.zeros((k, 3), dtype=int), np.zeros(k)
        k = min(k, len(pixels))
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels, minlength=k)
        weights = counts / counts.sum()
        return centers, weights

    for idx, piece in enumerate(pieces):
        contours = piece.get_contours()

        M = cv2.moments(np.array(contours))
        cx, cy = 0, 0
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

        top = bottom = left = right = None
        t = 3
        circle_radius = 100

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

        mask = np.zeros(sides_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [np.array(contours)], -1, 255, -1)

        top_side, bottom_side, left_side, right_side = [], [], [], []
        height, width = mask.shape
        sampling_stride = 3
        MAX_SAMPLES = 1000

        for y in range(0, height, sampling_stride):
            for x in range(0, width, sampling_stride):
                if mask[y, x] == 0:
                    continue

                pt = np.array([x, y])
                color = img[y, x]

                if top is not None and np.linalg.norm(pt - np.array(top)) <= circle_radius:
                    top_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (0, 0, 255), -1)
                elif bottom is not None and np.linalg.norm(pt - np.array(bottom)) <= circle_radius:
                    bottom_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (0, 255, 0), -1)
                elif left is not None and np.linalg.norm(pt - np.array(left)) <= circle_radius:
                    left_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (255, 0, 0), -1)
                elif right is not None and np.linalg.norm(pt - np.array(right)) <= circle_radius:
                    right_side.append(color)
                    if window:
                        cv2.circle(sides_img, (x, y), 1, (0, 255, 255), -1)

        top_side = top_side[:MAX_SAMPLES]
        bottom_side = bottom_side[:MAX_SAMPLES]
        left_side = left_side[:MAX_SAMPLES]
        right_side = right_side[:MAX_SAMPLES]

        top_colors, top_weights = get_multiple_dominant_colors(top_side, k=k_dominant_colors)
        bottom_colors, bottom_weights = get_multiple_dominant_colors(bottom_side, k=k_dominant_colors)
        left_colors, left_weights = get_multiple_dominant_colors(left_side, k=k_dominant_colors)
        right_colors, right_weights = get_multiple_dominant_colors(right_side, k=k_dominant_colors)

        def ensure_bgr(color):
            return (int(color[0]), int(color[1]), int(color[2]))

        x_offset = 0
        y_offset = 0
        color_width = 20
        color_height = 40

        def draw_colors(colors, weights, x_off):
            for i, (color, weight) in enumerate(zip(colors, weights)):
                bgr = ensure_bgr(color)
                y = y_offset + i * (color_height + 5)
                if window:
                    cv2.rectangle(sides_img, (x_off, y), (x_off + color_width, y + color_height), bgr, -1)
                    cv2.putText(sides_img, f"{weight:.2f}", (x_off + color_width + 5, y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        draw_colors(top_colors, top_weights, x_offset)
        draw_colors(bottom_colors, bottom_weights, x_offset + 60)
        draw_colors(left_colors, left_weights, x_offset + 120)
        draw_colors(right_colors, right_weights, x_offset + 180)

        if window:
            cv2.imshow(f"Piece {idx} - Dominant Colors", sides_img)
            cv2.waitKey(0)

        # Now assign (color, weight) tuple to each side
        s1, s2, s3, s4 = piece.get_sides()
        s1.set_side_color(left_colors, left_weights)
        s2.set_side_color(bottom_colors, bottom_weights)
        s3.set_side_color(right_colors, right_weights)
        s4.set_side_color(top_colors, top_weights)

    if window:
        cv2.destroyAllWindows()

    print("Color found..")
    return

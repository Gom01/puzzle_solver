import os
import warnings

from sklearn.exceptions import ConvergenceWarning

os.environ['OMP_NUM_THREADS'] = '1'

def find_color(puzzle, window=True, k_dominant_colors=3):
    from sklearn.cluster import KMeans
    import numpy as np
    import cv2


    pieces = puzzle.get_pieces()

    def get_dominant_colors(pixels, k):

        if len(pixels) == 0:
            return np.zeros((k, 3), dtype=int), np.zeros(k)
        pixels = np.array(pixels)
        k = min(k, len(pixels))
        kmeans = KMeans(n_clusters=k, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels)
        weights = counts / counts.sum()
        return centers, weights

    def sample_pixels_around_center(img, center, radius):
        pixels = []
        cx, cy = center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    bgr_pixel = img[y, x]
                    if not np.all(bgr_pixel == [0, 0, 0]):  # Skip masked/black pixels
                        pixels.append(bgr_pixel)
        return pixels

    def sample_n_points_along_contour(contour, n_points):
        contour = np.array(contour, dtype=np.float32)
        filtered_contour = [contour[0]]
        for pt in contour[1:]:
            if np.linalg.norm(pt - filtered_contour[-1]) > 1e-5:
                filtered_contour.append(pt)
        contour = np.array(filtered_contour)
        if len(contour) < 2:
            raise ValueError("The contour must have at least two distinct points.")
        distances = [0]
        for i in range(1, len(contour)):
            d = np.linalg.norm(contour[i] - contour[i - 1])
            distances.append(distances[-1] + d)
        total_length = distances[-1]
        step = total_length / (n_points - 1)
        sampled_points = []
        for i in range(n_points):
            target_dist = i * step
            for j in range(1, len(distances)):
                segment_length = distances[j] - distances[j - 1]
                if segment_length == 0:
                    continue
                if distances[j] >= target_dist:
                    ratio = (target_dist - distances[j - 1]) / segment_length
                    point = (1 - ratio) * contour[j - 1] + ratio * contour[j]
                    sampled_points.append(tuple(map(int, point)))
                    break
        return sampled_points, step

    def interpolate_missing_colors(colors):
        result = colors.copy()
        for i, color in enumerate(colors):
            if color is None:
                prev = next((colors[j] for j in range(i - 1, -1, -1) if colors[j] is not None), None)
                next_ = next((colors[j] for j in range(i + 1, len(colors)) if colors[j] is not None), None)
                if prev is not None and next_ is not None:
                    result[i] = ((np.array(prev) + np.array(next_)) // 2).astype(int).tolist()
                elif prev is not None:
                    result[i] = prev
                elif next_ is not None:
                    result[i] = next_
                else:
                    result[i] = [0, 0, 0]
        return result

    for idx, piece in enumerate(pieces):
        if piece.get_corners()[0] == [2,2]:
            continue

        img = piece.get_color_image().copy()
        for side_idx, side in enumerate(piece.get_sides()):
            side_contour = side.get_side_contour()
            sampled_points, step = sample_n_points_along_contour(side_contour, n_points=10)
            radius = int(step / 2)
            dominant_colors_per_side = []

            for pt in sampled_points:
                x, y = pt
                colors = sample_pixels_around_center(img, (x, y), radius)
                if not colors:
                    dominant_colors_per_side.append(None)
                    continue
                dom_colors, weights = get_dominant_colors(colors, k_dominant_colors)
                max_weight_idx = np.argmax(weights)
                dominant_color = dom_colors[max_weight_idx]
                cv2.circle(img, (x, y), radius, tuple(int(c) for c in dominant_color.tolist()), -1)
                dominant_colors_per_side.append(dominant_color[::-1])

            dominant_colors_per_side = interpolate_missing_colors(dominant_colors_per_side)
            side.set_side_color2(dominant_colors_per_side)

        if window:
            cv2.imshow(f"Piece {idx + 1}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return puzzle

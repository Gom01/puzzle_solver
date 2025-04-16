import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import procrustes

def side_similarities(side1, side2):

    def normalize_curve(points):
        points = np.array(points)
        points -= np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist != 0:
            points /= max_dist
        return points

    def align_curves(curve1, curve2):
        # Both should already be normalized and resampled
        mtx1, mtx2, disparity = procrustes(curve1, curve2)
        return mtx1, mtx2, disparity

    def resample_curve(points, num_points=100):
        points = np.array(points)
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative = np.insert(np.cumsum(distances), 0, 0)
        total_length = cumulative[-1]
        if total_length == 0:
            return np.repeat(points[0:1], num_points, axis=0)
        uniform_dist = np.linspace(0, total_length, num_points)
        interp_x = interp1d(cumulative, points[:, 0], kind='linear')
        interp_y = interp1d(cumulative, points[:, 1], kind='linear')
        resampled = np.stack((interp_x(uniform_dist), interp_y(uniform_dist)), axis=-1)
        return resampled

    def compare_curves(curve1, curve2):
        curve1 = normalize_curve(resample_curve(curve1))
        curve2 = normalize_curve(resample_curve(curve2))
        aligned1, aligned2, disparity = align_curves(curve1, curve2)
        score = np.mean(np.linalg.norm(aligned1 - aligned2, axis=1))
        return score, aligned1, aligned2



    # Compare first side of both pieces
    contours1 = side1.get_side_contour()
    contours2 = side2.get_side_contour()

    score, curve1, curve2 = compare_curves(contours1, contours2)
    score = score*10
    print(f"🔍 Side1 - Side2 | Similarity Score: {score:.4f}")

    # # --- Plotting ---
    # plt.figure(figsize=(6, 6))
    # plt.plot(curve1[:, 0], curve1[:, 1], label='Side1', color='blue')
    # plt.plot(curve2[:, 0], curve2[:, 1], label='Side2', color='red', linestyle='--')
    # plt.title(f"Similarity Score: {score:.4f}")
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(True)
    # plt.show()

    return score





import itertools

def color_similarities(colors1, weights1, colors2, weights2):
    """
    Compare two tables of 5 RGB colors + weights and return a similarity score (0 to 1).
    Inputs:
        colors1, colors2: np.ndarray of shape (5, 3)
        weights1, weights2: np.ndarray of shape (5,)
    """
    MAX_DISTANCE = np.linalg.norm([255, 255, 255])

    def color_similarity(c1, c2):
        dist = np.linalg.norm(c1 - c2)
        return 1 - (dist / MAX_DISTANCE)  # normalized similarity

    def compare_color_tables(c1, w1, c2, w2):
        best_weighted_score = 0
        for perm_colors2, perm_weights2 in zip(itertools.permutations(c2), itertools.permutations(w2)):
            similarities = [color_similarity(ci1, ci2) for ci1, ci2 in zip(c1, perm_colors2)]
            combined_weights = [(wi1 + wi2) / 2 for wi1, wi2 in zip(w1, perm_weights2)]  # avg weight
            weighted_score = np.sum([sim * weight for sim, weight in zip(similarities, combined_weights)])
            best_weighted_score = max(best_weighted_score, weighted_score)
        return best_weighted_score

    score = compare_color_tables(colors1, weights1, colors2, weights2)
    print(f"🎨 Weighted Table Similarity Score: {score:.4f}")
    return score




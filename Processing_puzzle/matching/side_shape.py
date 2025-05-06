import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import procrustes



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




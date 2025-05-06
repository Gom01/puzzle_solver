import itertools
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
from math import sqrt



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


from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def color_similarities2(colors1, colors2, weight=1.0, window=False, match_threshold=0.9):
    length = min(len(colors1), len(colors2))
    if length == 0:
        return 0.0

    max_dist = sqrt(3 * 255 ** 2)
    colors1 = np.array(colors1)
    colors2 = np.array(colors2)

    # Build cost matrix of Euclidean distances
    cost_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            dist = np.linalg.norm(colors1[i] - colors2[j])
            cost_matrix[i, j] = dist

    # Solve the assignment problem (minimize total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    match_count = 0
    similarities = []
    for i, j in zip(row_ind, col_ind):
        dist = cost_matrix[i, j]
        sim = 1 - (dist / max_dist)
        similarities.append(sim)
        if sim >= match_threshold:
            match_count += 1

    normalized_score = (match_count / length) * weight

    if window:
        fig, ax = plt.subplots(figsize=(6, length * 0.5))
        for idx, (i, j) in enumerate(zip(row_ind, col_ind)):
            c1 = colors1[i] / 255
            c2 = colors2[j] / 255
            ax.barh(idx, 1, color=c1, height=0.4, label="Color 1" if idx == 0 else "")
            ax.barh(idx + 0.4, 1, color=c2, height=0.4, label="Color 2" if idx == 0 else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, length + 0.5)
        ax.set_yticks([i + 0.2 for i in range(length)])
        ax.set_yticklabels([f"Pair {i + 1}" for i in range(length)])
        ax.set_xticks([])
        ax.legend(loc="upper right")
        ax.set_title(f"Perfect Match Score (Best Alignment): {normalized_score:.4f}")
        plt.tight_layout()
        plt.show()

    return normalized_score




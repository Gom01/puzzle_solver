from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab  # <--- Add this

def color_similarities2(colors1, colors2, weight=1.0, window=False):
    length = min(len(colors1), len(colors2))
    if length == 0:
        return 0.0

    # Convert to numpy and normalize to [0, 1]
    colors1 = np.array(colors1) / 255.0
    colors2 = np.array(colors2) / 255.0

    # Convert to Lab
    colors1_lab = rgb2lab(colors1.reshape(1, -1, 3)).reshape(-1, 3)
    colors2_lab = rgb2lab(colors2.reshape(1, -1, 3)).reshape(-1, 3)

    # Max distance in Lab is not bounded like RGB; use empirical upper bound (~100)
    max_dist = 100.0

    # Build cost matrix of Lab distances
    cost_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            dist = np.linalg.norm(colors1_lab[i] - colors2_lab[j])
            cost_matrix[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    similarities = []
    for i, j in zip(row_ind, col_ind):
        dist = cost_matrix[i, j]
        sim = 1 - (dist / max_dist)
        similarities.append(sim)

    normalized_score = (sum(similarities) / length) * weight

    if window:
        fig, ax = plt.subplots(figsize=(6, length * 0.5))
        for idx, (i, j) in enumerate(zip(row_ind, col_ind)):
            c1 = colors1[i]
            c2 = colors2[j]
            ax.barh(idx, 1, color=c1, height=0.4, label="Color 1" if idx == 0 else "")
            ax.barh(idx + 0.4, 1, color=c2, height=0.4, label="Color 2" if idx == 0 else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, length + 0.5)
        ax.set_yticks([i + 0.2 for i in range(length)])
        ax.set_yticklabels([f"Pair {i + 1}" for i in range(length)])
        ax.set_xticks([])
        ax.legend(loc="upper right")
        ax.set_title(f"Lab Match Score (Best Alignment): {normalized_score:.4f}")
        plt.tight_layout()
        plt.show()

    return normalized_score

import numpy as np
import cv2
from Processing_puzzle import Puzzle as p
import matplotlib.pyplot as plt

# ---- Utility Functions ----

def parse_cnt(points, num=100):
    """Resample contour to have fixed number of points evenly spaced by arc length."""
    points = np.array(points, dtype=np.float64)
    if len(points) < 2:
        return points
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])
    normalized_distances = distances / distances[-1]
    target = np.linspace(0, 1, num)
    x = np.interp(target, normalized_distances, points[:, 0])
    y = np.interp(target, normalized_distances, points[:, 1])
    return np.stack([x, y], axis=1)

def normalize_contour_uniform(contour):
        contour = np.array(contour, dtype=np.float64)
        diffs = np.diff(contour, axis=0)
        total_length = np.sum(np.linalg.norm(diffs, axis=1))
        return contour / total_length if total_length > 0 else contour


def align_by_centroid_rotation(curve):
    """Center the curve at its centroid and rotate so main axis is horizontal."""
    curve = np.array(curve, dtype=np.float64)
    center = np.mean(curve, axis=0)
    curve_centered = curve - center
    vec = curve_centered[-1] - curve_centered[0]
    angle = -np.arctan2(vec[1], vec[0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = curve_centered @ R.T
    return rotated

def compute_distance(curve1, curve2):
    """Compute per-point Euclidean distance between two curves."""
    return np.linalg.norm(curve1 - curve2, axis=1)

def compute_score(distances):
    """Compute total squared distance score."""
    return np.sum(distances ** 2)

def plot_two_contours_with_score(curve1, curve2, label1="Curve 1", label2="Curve 2",
                                  color1="blue", color2="red", score=None):
    x1, y1 = zip(*curve1)
    x2, y2 = zip(*curve2)
    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, marker='o', linestyle='-', color=color1, label=label1)
    plt.plot(x2, y2, marker='o', linestyle='-', color=color2, label=label2)
    plt.title("Aligned Contours")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    if score is not None:
        plt.text(0.95, 0.95, f'Score: {score:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, ha='right', va='top', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.show()

# ---- Main Similarity Function ----

def side_similarities(side1, side2, window=False):
    """
    Compare two sides by resampling, normalizing, aligning and scoring.
    Tries both normal and reversed directions for side2 and returns best.
    """
    parsed_1 = parse_cnt(side1.get_side_contour())
    parsed_2 = parse_cnt(side2.get_side_contour())

    scores = []

    for variant in [parsed_2, parsed_2[::-1]]:
        norm_1 = normalize_contour_uniform(parsed_1)
        norm_2 = normalize_contour_uniform(variant)

        aligned_1 = align_by_centroid_rotation(norm_1)
        aligned_2 = align_by_centroid_rotation(norm_2)

        distances = compute_distance(aligned_1, aligned_2)
        score = compute_score(distances)
        scores.append((score, aligned_2))

    best_score, best_aligned_2 = min(scores, key=lambda x: x[0])
    confidence = 1 / (1 + best_score)

    if window:
        plot_two_contours_with_score(aligned_1, best_aligned_2, label1="Aligned Side 1", label2="Side 2", score=confidence)

    return confidence

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Processing_puzzle.matching.side_shape import best_cyclic_shift_alignment

def compute_curvature(contour):
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
    return curvature

def parse_cnt(points, num=100):
    points = np.array(points, dtype=np.float64)
    if len(points) < 2:
        return points
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])
    normalized_distances = distances / distances[-1]
    target = np.linspace(0, 1, num)
    x = np.interp(target, normalized_distances, points[:, 0])
    y = np.interp(target, normalized_distances, points[:, 1])
    return np.stack([x, y], axis=1)

def normalize_curve(curve):
    curve = curve - np.mean(curve, axis=0)
    norm = np.linalg.norm(curve)
    return curve / norm if norm > 0 else curve

def smooth_curve(curve, window_size=5):
    kernel = np.ones((window_size, 1)) / window_size
    x = np.convolve(curve[:, 0], kernel.ravel(), mode='same')
    y = np.convolve(curve[:, 1], kernel.ravel(), mode='same')
    return np.stack([x, y], axis=1)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def compare_curvature(side1, side2, window=False):
    # Preprocess: resample, normalize, smooth
    c1 = normalize_curve(smooth_curve(parse_cnt(side1.get_side_contour(), 100)))
    c2 = normalize_curve(smooth_curve(parse_cnt(side2.get_side_contour(), 100)))

    # Flip and align
    variants = [c2, c2[::-1]]
    best_score = float("inf")
    best_aligned = None
    best_variant = None

    for v in variants:
        score, aligned = best_cyclic_shift_alignment(c1, v)
        if score < best_score:
            best_score = score
            best_aligned = aligned
            best_variant = v

    # Compute curvature and smooth it
    k1 = gaussian_filter1d(compute_curvature(c1), sigma=2)
    k2 = gaussian_filter1d(compute_curvature(best_aligned), sigma=2)

    # Similarity via cosine similarity
    similarity = max(0.0, min(1.0, cosine_similarity(k1, k2)))

    # Optional plot
    if window:
        x = np.linspace(0, 1, len(k1))

        plt.figure(figsize=(12, 5))

        # Curvature profile comparison
        plt.subplot(1, 2, 1)
        plt.plot(x, k1, label='Curvature Side 1', color='blue')
        plt.plot(x, k2, label='Curvature Side 2 (aligned)', color='red')
        plt.fill_between(x, k1, k2, color='gray', alpha=0.3, label='Difference')
        plt.title("Curvature Profiles")
        plt.xlabel("Normalized Arc Length")
        plt.ylabel("Curvature")
        plt.grid(True)
        plt.legend()

        # Contour visualization
        plt.subplot(1, 2, 2)
        plt.plot(c1[:, 0], c1[:, 1], label="Side 1", color="blue")
        plt.plot(best_variant[:, 0], best_variant[:, 1], label="Side 2 (pre-alignment)", color="red", alpha=0.6)
        plt.plot(best_aligned[:, 0], best_aligned[:, 1], label="Side 2 (aligned)", color="green", linestyle='--')
        plt.gca().invert_yaxis()
        plt.title(f"Sides Shape (Similarity: {similarity:.3f})")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return similarity

import numpy as np
import matplotlib.pyplot as plt

# ---------- Utility Functions ----------

def parse_cnt(points, num=100):
    """Resample contour to have a fixed number of points evenly spaced by arc length."""
    points = np.array(points, dtype=np.float64)
    if len(points) < 2:
        return points
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])
    normalized_distances = distances / distances[-1]
    target = np.linspace(0, 1, num)
    x = np.interp(target, normalized_distances, points[:, 0])
    y = np.interp(target, normalized_distances, points[:, 1])
    return np.stack([x, y], axis=1)

def align_contour_to_side_axis(contour):
    """Translate contour to origin and rotate so its direction lies along the X-axis."""
    contour = np.array(contour, dtype=np.float64)
    start = contour[0]
    end = contour[-1]
    direction = end - start
    angle = -np.arctan2(direction[1], direction[0])
    translated = contour - start
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    aligned = translated @ R.T
    return aligned

def normalize_curve(curve):
    """Translate to origin and normalize to unit length."""
    centered = curve - np.mean(curve, axis=0)
    scale = np.linalg.norm(centered)
    return centered / scale if scale > 0 else centered

def compute_curvature(contour):
    """Return the curvature along a contour (approximated)."""
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
    return curvature

def soft_hausdorff_distance(A, B):
    """Compute a bidirectional average min distance (soft Hausdorff)."""
    dists_a_to_b = np.min(np.linalg.norm(A[:, None] - B[None, :], axis=2), axis=1)
    dists_b_to_a = np.min(np.linalg.norm(B[:, None] - A[None, :], axis=2), axis=1)
    return np.mean(np.concatenate([dists_a_to_b, dists_b_to_a]))

def best_cyclic_shift_alignment(curve1, curve2):
    """Try all circular shifts of curve2 to best match curve1 using soft Hausdorff."""
    best_score = float("inf")
    best_curve2 = curve2
    for shift in range(len(curve2)):
        shifted = np.roll(curve2, shift, axis=0)
        score = soft_hausdorff_distance(curve1, shifted)
        if score < best_score:
            best_score = score
            best_curve2 = shifted
    return best_score, best_curve2

def plot_two_contours_with_score(curve1, curve2, label1="Side 1", label2="Side 2",
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
        plt.text(0.95, 0.95, f'Score: {score:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, ha='right', va='top', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.show()

# ---------- Main Function ----------

def side_similarities(side1, side2, window=False):
    """
    Compare two sides by aligning to the same axis, flipping/reversing, and scoring.
    Now robust to zoomed pieces.
    """
    # Parse and normalize both sides
    parsed_1 = normalize_curve(align_contour_to_side_axis(parse_cnt(side1.get_side_contour(), num=100)))
    parsed_2_raw = parse_cnt(side2.get_side_contour(), num=100)

    best_score = float("inf")
    best_curve2 = None

    # Try all flips/reversals/negations
    for variant in [parsed_2_raw, parsed_2_raw[::-1], -parsed_2_raw, -parsed_2_raw[::-1]]:
        aligned_variant = normalize_curve(align_contour_to_side_axis(variant))
        score, shifted = best_cyclic_shift_alignment(parsed_1, aligned_variant)
        if score < best_score:
            best_score = score
            best_curve2 = shifted

    alpha = 10  # You can tweak this (e.g., 3.0, 5.0, 7.0)
    confidence = np.exp(-alpha * best_score)

    # Optional visualization
    if window:
        plot_two_contours_with_score(parsed_1, best_curve2, score=confidence)

    return confidence


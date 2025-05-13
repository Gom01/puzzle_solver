import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def compute_normals(contour):
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    tangents = np.stack([dx, dy], axis=1)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([-dy, dx], axis=1) / (norms + 1e-8)
    return normals

def sample_points_inward(contour, normals, radius=3):
    return [tuple(map(int, pt + radius * n)) for pt, n in zip(contour, normals)]

def get_gradient_at_points(image, points):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitudes, angles = [], []
    for x, y in points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            gx, gy = grad_x[y, x], grad_y[y, x]
            magnitudes.append(np.hypot(gx, gy))
            angles.append(np.arctan2(gy, gx))
    return np.array(magnitudes), np.array(angles)

def gradient_histogram(angles, magnitudes=None, bins=16):
    if magnitudes is None:
        magnitudes = np.ones_like(angles)
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=magnitudes)
    return hist / (np.sum(hist) + 1e-8)

def compare_histograms(hist1, hist2):
    return np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-8)

def rotate_image_and_contour(image, contour, angle_deg):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    contour = np.array(contour)
    ones = np.ones((len(contour), 1))
    pts = np.hstack([contour, ones])
    rotated_contour = pts @ M.T
    return rotated_image, rotated_contour

def gradient_descriptor_for_side(image, contour, num_points=50, inward_radius=3, bins=16):
    contour = parse_cnt(contour, num_points)
    normals = compute_normals(contour)
    points = sample_points_inward(contour, normals, radius=inward_radius)
    magnitudes, angles = get_gradient_at_points(image, points)
    hist = gradient_histogram(angles, magnitudes, bins=bins)
    return hist, points, magnitudes, angles, image

def compare_side_gradients(side1, side2, bins=16, window=False):
    # Base side 1
    img1 = side1.get_piece_image().copy()
    cnt1 = side1.get_side_contour()
    hist1, pts1, mags1, angs1, _ = gradient_descriptor_for_side(img1, cnt1, bins=bins)

    best_score = -1
    best_angle = 0
    best_data = None

    for angle in [0, 90, 180, 270]:
        img2 = side2.get_piece_image().copy()
        cnt2 = side2.get_side_contour()
        rotated_img2, rotated_cnt2 = rotate_image_and_contour(img2, cnt2, angle)
        hist2, pts2, mags2, angs2, _ = gradient_descriptor_for_side(rotated_img2, rotated_cnt2, bins=bins)
        score = compare_histograms(hist1, hist2)

        if score > best_score:
            best_score = score
            best_angle = angle
            best_data = (rotated_img2, rotated_cnt2, pts2, mags2, angs2)

    if window:
        vis1 = img1.copy()
        for (x, y), mag, angle in zip(pts1, mags1, angs1):
            pt1 = (int(x), int(y))
            pt2 = (int(x + np.cos(angle) * 2), int(y + np.sin(angle) * 2))
            cv2.arrowedLine(vis1, pt1, pt2, (0, 0, 255), 1, tipLength=0.3)
        vis1_rgb = cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB)

        img2, _, pts2, mags2, angs2 = best_data
        vis2 = img2.copy()
        for (x, y), mag, angle in zip(pts2, mags2, angs2):
            pt1 = (int(x), int(y))
            pt2 = (int(x + np.cos(angle) * 2), int(y + np.sin(angle) * 2))
            cv2.arrowedLine(vis2, pt1, pt2, (0, 0, 255), 1, tipLength=0.3)
        vis2_rgb = cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(vis1_rgb)
        axs[0].set_title("Side 1 Gradients")
        axs[0].axis('off')
        axs[1].imshow(vis2_rgb)
        axs[1].set_title(f"Side 2 Gradients (Rotated {best_angle}°)")
        axs[1].axis('off')
        plt.suptitle(f"Gradient Similarity Score: {best_score:.4f}")
        plt.tight_layout()
        plt.show()

    return best_score

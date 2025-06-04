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

    def compute_black_pixel_ratio_on_line(
            piece, pt1, pt2, infoPS, offset=5, n_bins=30, seuil=35
    ):
        import numpy as np
        import cv2
        from skimage.draw import line as skimage_line
        import matplotlib.pyplot as plt

        contours = piece.get_contours()
        contour_np = np.array(contours, dtype=np.int32).reshape((-1, 1, 2))
        moments = cv2.moments(contour_np)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        img2 = piece.get_color_image()

        pt1 = np.array(pt1, dtype=float)
        pt2 = np.array(pt2, dtype=float)
        v = pt2 - pt1
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return 0.0

        vp = np.array([-v[1], v[0]]) / v_norm
        mx, my = (pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2
        vec_c = np.array([cx - mx, cy - my])
        if np.dot(vp, vec_c) < 0:
            vp = -vp

        pt1_in = pt1 + offset * vp
        pt2_in = pt2 + offset * vp
        x1, y1 = int(round(pt1_in[0])), int(round(pt1_in[1]))
        x2, y2 = int(round(pt2_in[0])), int(round(pt2_in[1]))
        x1 = np.clip(x1, 0, img2.shape[1] - 1)
        x2 = np.clip(x2, 0, img2.shape[1] - 1)
        y1 = np.clip(y1, 0, img2.shape[0] - 1)
        y2 = np.clip(y2, 0, img2.shape[0] - 1)

        rr, cc = skimage_line(y1, x1, y2, x2)
        rr = np.clip(rr, 0, img2.shape[0] - 1)
        cc = np.clip(cc, 0, img2.shape[1] - 1)
        line_pixels = img2[rr, cc]
        N = len(rr)
        if N == 0:
            return 0.0

        is_black = np.all(line_pixels == [0, 0, 0], axis=1).astype(int)
        pct_black = 100 * np.sum(is_black) / N

        bin_indices = np.linspace(0, N, n_bins + 1, dtype=int)
        black_bin_vals = []
        for i in range(n_bins):
            seg = is_black[bin_indices[i]:bin_indices[i + 1]]
            if len(seg) == 0:
                black_bin_vals.append(np.nan)
            else:
                black_bin_vals.append(np.mean(seg))
        black_bin_vals = np.array(black_bin_vals)

        # Correction : on ignore le premier et le dernier bin pour la détection d'anomalie
        black_bin_vals_mid = black_bin_vals[1:-1]
        black_bin_vals_display = np.copy(black_bin_vals)
        black_bin_vals_display[black_bin_vals_display < 0.6] = 0
        black_bin_vals_mid_display = black_bin_vals_display[1:-1]

        # ----------- Analyse fragmentation intégrée sur les bins centraux -----------
        binaire = (black_bin_vals_mid_display >= 0.6).astype(int)
        transitions = np.diff(binaire, prepend=0)
        nb_segments_noirs = np.sum(transitions == 1)

        show_R = False

        if pct_black > 50:
            print(f"{infoPS} ANORMAL - {pct_black} % de Noir (analyse sans 1er/dernier bin)")
            piece.is_bad = True
            show_R = True

        elif nb_segments_noirs > 1 and pct_black > seuil:
            print(f"{infoPS} ANORMAL - {nb_segments_noirs} segments noirs séparés (analyse sans 1er/dernier bin)")
            piece.set_wrong_sides()
            show_R = True

        # ---------------------------------------------------------------------------
        if show_R:
            print(f"{infoPS} - {pct_black} % de Noir (analyse sans 1er/dernier bin)")


        val_majoritaire = int(np.round(black_bin_vals.mean() > 0.5))

        img_disp = img2.copy()
        for y, x in zip(rr, cc):
            img_disp[y, x] = (255, 0, 255)
        cv2.line(img_disp, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        cv2.circle(img_disp, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        if show_R :
            cv2.imshow(f"{infoPS} - {pct_black:.2f}% noirs", img_disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if show_R:
            x_normalized = np.linspace(0, 1, n_bins)
            colors = ['red'] + ['black'] * (n_bins - 2) + ['red']  # Bins ignorés en rouge

            plt.figure(figsize=(8, 3))
            bars = plt.bar(x_normalized, black_bin_vals_display, width=1 / n_bins, color=colors, alpha=0.6)

            # Ajout de la légende personnalisée
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', label='Bins analysés'),
                Patch(facecolor='red', label='Bins ignorés')
            ]
            plt.legend(handles=legend_elements, loc='best')

            plt.axhline(0.5, color='r', linestyle='--', alpha=0.3)
            # Ligne centroïde projeté
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            centroid = np.array([cx, cy])
            dp = p2 - p1
            dp_norm = dp / np.linalg.norm(dp)
            v_centroid = centroid - p1
            proj_length = np.dot(v_centroid, dp_norm)
            total_length = np.linalg.norm(p2 - p1)
            pos_norm = np.clip(proj_length / total_length, 0, 1)
            plt.axvline(pos_norm, color="blue", linestyle="--", label="Centroïde projeté")

            plt.title(
                f"{infoPS} Profil de noirceur ({n_bins} bins) — majorité : {val_majoritaire} "
                f"({'noir' if val_majoritaire == 1 else 'non-noir'})"
            )
            plt.xlabel("Distance normalisée sur la droite (pt1 → pt2)")
            plt.ylabel("Fréquence de pixels noirs")
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.show()

        return

    for idx, piece in enumerate(pieces):
        if piece.get_corners()[0] == [2,2]:
            continue

        img = piece.get_color_image().copy()
        for side_idx, side in enumerate(piece.get_sides()):
            side_contour = side.get_side_contour()

            pt1 = side_contour[0]
            pt2 = side_contour[-1]
            infoPS = f"P:{idx + 1},C:{side_idx + 1}"
            compute_black_pixel_ratio_on_line(piece, pt1, pt2,infoPS, offset=5)


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
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    return puzzle

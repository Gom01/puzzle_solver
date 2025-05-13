import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt


def afficher_contour_polaire(contour_np, percentile_seuil=80, distance_tolerance=0.5):
    """
    Affiche un contour en coordonnées cartésiennes et polaires, et filtre uniquement les extrêmes locaux pointus,
    avec des critères de filtrage ajustés pour permettre un meilleur affichage.
    """
    if contour_np.ndim != 3 or contour_np.shape[1:] != (1, 2):
        raise ValueError("Le contour doit avoir la forme (N, 1, 2)")

    # Aplatir le contour (N, 2)
    points = contour_np.reshape(-1, 2)

    # Calcul du centroïde
    cx, cy = points.mean(axis=0)

    # Conversion en coordonnées polaires
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    rho = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    # --- Calcul de l'angle moyen entre les points voisins ---
    angles = np.diff(theta)
    angles = np.concatenate([angles, [angles[0]]])  # Retourner à l'angle initial (circularité)

    # Calcul de l'angle moyen
    mean_angle = np.mean(np.abs(angles))
    print(f"Angle moyen : {mean_angle} radians")

    # Plage dynamique d'angles
    min_angle = 20  # Fixer un seuil minimum réaliste pour l'angle
    max_angle = 90  # Fixer un seuil maximum réaliste pour l'angle
    print(f"Plage d'angles dynamique : {min_angle}° à {max_angle}°")

    # --- Filtrage des pics locaux ---
    rho_derivative = np.diff(rho)
    local_max_indices = np.where((rho_derivative[:-1] > 0) & (rho_derivative[1:] < 0))[0] + 1  # Indices des pics locaux

    # Vérifier les pics locaux trouvés
    print(f"Nombre de pics locaux trouvés avant le filtrage : {len(local_max_indices)}")

    # --- Filtrage basé sur l'angle ---
    valid_angles = (angles > np.radians(min_angle)) & (angles < np.radians(max_angle))
    valid_local_max_indices = local_max_indices[valid_angles[local_max_indices]]
    print(f"Nombre de pics locaux après filtrage des angles : {len(valid_local_max_indices)}")

    # --- Calcul de la courbure ---
    rho_second_derivative = np.diff(rho_derivative)
    curvatures = np.abs(rho_second_derivative)

    # Affichage de la distribution des courbures pour ajuster le percentile
    plt.figure(figsize=(8, 4))
    plt.hist(curvatures, bins=20, color='g', alpha=0.7)
    plt.title('Distribution des courbures')
    plt.xlabel('Courbure')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()

    # Afficher les courbures et les indices pour chaque point
    for i, curv in enumerate(curvatures):
        print(f"Index {i}, Courbure : {curv}")

    # Filtrer les indices avec une courbure élevée
    threshold = np.percentile(curvatures, percentile_seuil)
    sharp_max_indices = valid_local_max_indices[curvatures[valid_local_max_indices - 1] > threshold]
    print(f"Nombre de pics locaux après filtrage de la courbure (seuil {percentile_seuil}%) : {len(sharp_max_indices)}")

    # --- Filtrage basé sur la distance au centroïde ---
    max_distance = np.max(rho)
    distance_tolerance_indices = np.where(np.abs(rho - max_distance) / max_distance < distance_tolerance)[0]
    final_indices = np.intersect1d(sharp_max_indices, distance_tolerance_indices)
    print(f"Nombre de pics après filtrage basé sur la distance au centroïde : {len(final_indices)}")

    # Points filtrés
    filtered_points = points[final_indices]
    filtered_rho = rho[final_indices]
    filtered_theta = theta[final_indices]

    # --- Affichage cartésien ---
    plt.figure(figsize=(6, 6))
    plt.plot(points[:, 0], points[:, 1], 'o-', label='Contour')
    plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'ro', label='Extrêmes locaux pointus')
    plt.plot(cx, cy, 'rx', label='Centroïde')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title("Contour en coordonnées cartésiennes (Extrêmes locaux pointus)")
    plt.legend()
    plt.grid(True)

    # --- Affichage polaire (θ, ρ) ---
    plt.figure(figsize=(8, 5))
    plt.scatter(theta, rho, s=5, color='blue', label='Contour')
    plt.scatter(filtered_theta, filtered_rho, s=10, color='red', label='Extrêmes locaux pointus')
    plt.title("Représentation polaire du contour (θ, ρ) - Extrêmes locaux pointus")
    plt.xlabel("θ (radians)")
    plt.ylabel("ρ (distance au centroïde)")
    plt.grid(True)
    plt.legend()

    plt.show()


def find_corners(myPuzzle, window=False):
    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):
        colored_img = piece.get_color_image()
        colored_img = colored_img.copy()
        colored_img2 = colored_img.copy()
        colored_img3 = colored_img.copy()


        contours = piece.get_contours()

        contour_np = np.array(contours, dtype=np.int32).reshape((-1, 1, 2))

        moments = cv.moments(contour_np)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        epsilon = 0.005 * cv.arcLength(contour_np, True)
        approx = cv.approxPolyDP(contour_np, epsilon, True)
        hull = cv.convexHull(approx)

        points = [tuple(pt[0]) for pt in hull]

        def distance_point_to_line(x1, y1, x2, y2, cx, cy):
            numerator = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
            denominator = math.hypot(y2 - y1, x2 - x1)
            return numerator / denominator if denominator != 0 else float('inf')

        def get_points_through_centroid(points, cx, cy, radius=40):
            valid_points = set()
            for i, p1 in enumerate(points):
                x1, y1 = p1
                for j, p2 in enumerate(points):
                    if i >= j:
                        continue
                    x2, y2 = p2
                    dist = distance_point_to_line(x1, y1, x2, y2, cx, cy)
                    if dist <= radius:
                        valid_points.add(p1)
                        valid_points.add(p2)
            return list(valid_points)

        important_points = get_points_through_centroid(points, cx, cy, radius=40)

        def calculate_centroid(points):
            x = sum([pt[0] for pt in points]) / len(points)
            y = sum([pt[1] for pt in points]) / len(points)
            return (x, y)

        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(cos_theta) * 180.0 / np.pi
            return angle

        def calculate_area(quad):
            x1, y1 = quad[0]
            x2, y2 = quad[1]
            x3, y3 = quad[2]
            x4, y4 = quad[3]
            return 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - y1 * x2 - y2 * x3 - y3 * x4 - y4 * x1)

        def calculate_moment_of_inertia(quad, centroid):
            moment = 0
            for point in quad:
                distance = np.linalg.norm(np.array(point) - np.array(centroid))
                moment += distance ** 2
            return moment

        def check_side_lengths(quad):
            lengths = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                length = np.linalg.norm(np.array(p1) - np.array(p2))
                lengths.append(length)
            max_length = max(lengths)
            min_length = min(lengths)
            mean_length = np.mean(lengths)
            return (max_length - min_length) / mean_length <= 0.2

        def is_valid_quadrilateral(quad):
            angles = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                p3 = quad[(i + 2) % 4]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)
            return all(80 <= angle <= 99 for angle in angles)

        from itertools import combinations

        def find_best_corners(points, image):
            best_quads = []
            for quad in combinations(points, 4):
                centroid_quad = calculate_centroid(quad)
                if not is_valid_quadrilateral(quad):
                    continue
                if not check_side_lengths(quad):
                    continue
                moment = calculate_moment_of_inertia(quad, centroid_quad)
                area = calculate_area(quad)
                best_quads.append((quad, centroid_quad, moment, area))
            best_quads.sort(key=lambda x: x[3])
            if len(best_quads) > 0:
                best_quad = best_quads[0][0]
                for point in best_quad:
                    cv.circle(image, point, 8, (0, 255, 0), -1)
                return image, list(best_quad)
            return image, []

        image, filtered_points2 = find_best_corners(points, colored_img.copy())

        points = []
        if (len(filtered_points2) == 4):
            points = filtered_points2
        else:
            def dist(p1, p2):
                return int(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

            threshold = 1
            points_middle = important_points.copy()
            for i in range(0, len(important_points)):
                A = important_points[i]
                for j in range(0, len(important_points)):
                    B = important_points[j]
                    for k in range(0, len(important_points)):
                        C = important_points[k]
                        if A != B and A != C and C != B:
                            if (dist(A, B) + dist(B, C)) - threshold <= dist(A, C) <= (dist(A, B) + dist(B, C)) + threshold:
                                if B in points_middle:
                                    points_middle.remove(B)

            def order_points_by_angle(all_points):
                points = np.array(all_points)
                centroid = np.mean(points, axis=0)

                def calculate_angle(point):
                    dx, dy = point[0] - centroid[0], point[1] - centroid[1]
                    return np.arctan2(dy, dx)

                angles = [calculate_angle(pt) for pt in points]
                sorted_points = [pt for _, pt in sorted(zip(angles, points), key=lambda x: x[0])]
                return sorted_points

            def remove_close_points(all_points, image):
                ordered_points = order_points_by_angle(all_points)
                points = np.array(ordered_points)
                pairwise_distances = []
                for i in range(len(points) - 1):
                    distance = np.linalg.norm(points[i] - points[i + 1])
                    pairwise_distances.append(distance)
                distance = np.linalg.norm(points[len(points) - 1] - points[0])
                pairwise_distances.append(distance)
                max_distance = np.max(pairwise_distances)
                threshold = max_distance / 1.3
                filtered_points = []
                for i, point in enumerate(points):
                    if i > 0:
                        distance_to_prev = np.linalg.norm(point - points[i - 1])
                    else:
                        distance_to_prev = float('inf')
                    if i < len(points) - 1:
                        distance_to_next = np.linalg.norm(point - points[i + 1])
                    else:
                        distance_to_next = float('inf')
                    if distance_to_prev >= threshold or distance_to_next >= threshold:
                        filtered_points.append(tuple(point))
                return filtered_points, image

            filtered_points, image_result = remove_close_points(points_middle, colored_img2)
            colored_img = image

            contour_np = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

            def get_main_corners_from_min_rect(contour, search_radius=20):
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                refined_corners = []
                for corner in box:
                    dists = np.linalg.norm(contour.reshape(-1, 2) - corner, axis=1)
                    closest_pt = contour.reshape(-1, 2)[np.argmin(dists)]
                    refined_corners.append(tuple(closest_pt))
                return refined_corners

            points = get_main_corners_from_min_rect(contour_np)

        def order_corners(corners, cx, cy):
            def calculate_angle(pt):
                dx, dy = pt[0] - cx, pt[1] - cy
                return np.arctan2(dy, dx)
            sorted_corners = sorted(corners, key=calculate_angle)
            top_left = sorted_corners[0]
            bottom_left = sorted_corners[1]
            bottom_right = sorted_corners[2]
            top_right = sorted_corners[3]
            return [top_left, top_right, bottom_right, bottom_left]

        if (points[0] == points[1]) or (points[0] == points[2]) or (points[0] == points[3]) or (points[2] == points[3]):
            print(f"Corners of piece number {piece.index} are incorrect")
            piece.corners = [[-1, -1], [-1, -1], [-1, 1], [1, -1]]
        else:
            ordered_points = order_corners(points, cx, cy)
            piece.corners = ordered_points

            # === SHOW FINAL CORNERS IF window=True ===
            if window:
                final_img = colored_img3.copy()
                for pt in ordered_points:
                    cv.circle(final_img, pt, 10, (0, 255, 255), -1)  # Yellow corners
                cv.imshow(f"Final Corners - Piece {piece.index}", final_img)
                cv.waitKey(0)
                cv.destroyWindow(f"Final Corners - Piece {piece.index}")

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Corners saved...")
    return ()

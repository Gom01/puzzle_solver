from itertools import combinations

import cv2 as cv
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from Processing_puzzle.Puzzle import Puzzle as p

from itertools import combinations

'''
    Function findCorners : Find the 4 corners of a piece [(x,y),(x,y)...]
    Input: myPuzzle
'''





import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


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


def find_corners(myPuzzle):
    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):

        # Get basics information and conversion
        colored_img = piece.get_color_image()
        colored_img = colored_img.copy()
        colored_img2 = colored_img.copy()
        colored_img3 = colored_img.copy()

        contours = piece.get_contours()
        contour_np = np.array(contours, dtype=np.int32).reshape((-1, 1, 2))
        print("contour :",contour_np)
        #afficher_contour_polaire(contour_np)

        # Centroid of the piece
        moments = cv.moments(contour_np)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        # print("cx - cy :",cx, cy)

        # 1st approach : Draw basic polygon arround my piece to get the most important points
        epsilon = 0.005 * cv.arcLength(contour_np, True)
        approx = cv.approxPolyDP(contour_np, epsilon, True)
        hull = cv.convexHull(approx)

        # #Goal always have 4 points
        # cv.polylines(colored_img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in hull:
            cv.circle(colored_img, tuple(pt[0]), 6, (0, 255, 0), -1)
        cv.circle(colored_img, (cx, cy), 7, (0, 0, 255), -1)
        # cv.imshow("Important points", colored_img)
        # cv.waitKey(0)

        image = colored_img.copy()

        # 2nd approach : using the centroid removes all the points which are not symetric
        def distance_point_to_line(x1, y1, x2, y2, cx, cy):
            """Returns the perpendicular distance from point (cx, cy) to the line through (x1,y1)-(x2,y2)."""
            numerator = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
            denominator = math.hypot(y2 - y1, x2 - x1)
            return numerator / denominator if denominator != 0 else float('inf')

        def get_points_through_centroid(points, cx, cy, radius=40):
            """Returns a list of unique points where lines between them pass within radius of centroid."""
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

        points = [tuple(pt[0]) for pt in hull]
        important_points = get_points_through_centroid(points, cx, cy, radius=40)
        # print("points: ", points)
        # print("Important points: ", important_points)
        # print("\n\n\n\n")

        for point in important_points:
            cv.circle(image, point, 6, (0, 0, 255), -1)
            converted_point = (int(point[0]), int(point[1]))
            cv.putText(image, f"${converted_point}", (point[0] + 5, point[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 0), 1, cv.LINE_AA)

        # cv.imshow("Important points", colored_img)
        # cv.waitKey(0)

        # Fonction pour calculer le centroïde
        def calculate_centroid(points):
            x = sum([pt[0] for pt in points]) / len(points)
            y = sum([pt[1] for pt in points]) / len(points)
            return (x, y)

        # Fonction pour calculer l'angle entre trois points
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(cos_theta) * 180.0 / np.pi  # Conversion en degrés
            return angle

        # Fonction pour calculer l'aire d'un quadrilatère à partir des coordonnées de ses points
        def calculate_area(quad):
            x1, y1 = quad[0]
            x2, y2 = quad[1]
            x3, y3 = quad[2]
            x4, y4 = quad[3]
            return 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - y1 * x2 - y2 * x3 - y3 * x4 - y4 * x1)

        # Fonction pour calculer le moment d'inertie (lorsque les points sont éloignés du centroïde)
        def calculate_moment_of_inertia(quad, centroid):
            moment = 0
            for point in quad:
                distance = np.linalg.norm(np.array(point) - np.array(centroid))
                moment += distance ** 2  # Moment d'inertie (sommer les carrés des distances)
            return moment

        # Fonction pour vérifier que les longueurs des côtés sont à ±20% similaires
        def check_side_lengths(quad):
            # print(f"quad {quad}")
            lengths = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                length = np.linalg.norm(np.array(p1) - np.array(p2))
                # print(f"side length ${i}:",length)
                lengths.append(length)

            max_length = max(lengths)
            min_length = min(lengths)
            mean_length = np.mean(lengths)
            # print(f"quad {quad}, length ${i}:", (max_length - min_length) / mean_length,"\n")
            # Vérifier si la différence entre la longueur maximale et minimale est inférieure à 20% de la longueur moyenne
            return (max_length - min_length) / mean_length <= 0.2

        # Fonction de vérification des angles
        def is_valid_quadrilateral(quad):
            # print(f"quad {quad}")
            angles = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                p3 = quad[(i + 2) % 4]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)

            #   print(f"angle ${i}:",angle)
            # print("\n")
            # Vérification que tous les angles sont compris entre 85° et 95°
            return all(80 <= angle <= 99 for angle in angles)

        # Fonction principale pour trouver le meilleur quadrilatère
        def find_best_corners(points, image):
            best_quads = []

            # Vérification de tous les quadruplets possibles
            for quad in combinations(points, 4):

                # Calcul du centroïde du quadrilatère
                centroid_quad = calculate_centroid(quad)

                # Vérification des angles
                if not is_valid_quadrilateral(quad):
                    continue  # Si les angles sont invalides, on passe au quadrilatère suivant

                # Vérification des longueurs des côtés
                if not check_side_lengths(quad):
                    continue  # Si les longueurs des côtés ne sont pas similaires, on passe au quadrilatère suivant

                # Calcul du moment d'inertie
                moment = calculate_moment_of_inertia(quad, centroid_quad)

                # Calcul de l'aire du quadrilatère
                area = calculate_area(quad)

                # print(f"Moment d'inertie: {moment}")
                # print(f"Aire: {area}")

                # Ajouter au tableau si les critères sont remplis
                best_quads.append((quad, centroid_quad, moment, area))

            # Trier les quadrilatères par aire (du plus petit au plus grand)
            best_quads.sort(key=lambda x: x[3])

            if len(best_quads) > 0:
                # Retenir le quadrilatère avec l'aire la plus petite
                best_quad = best_quads[0][0]

                # Dessiner les points du meilleur quadrilatère
                for point in best_quad:
                    cv.circle(image, point, 8, (0, 255, 0), -1)  # Vert

                return image, list(best_quad)  # Retourner l'image et les coins du quadrilatère

            return image, []  # Aucun quadrilatère valide

        image, filtered_points2 = find_best_corners(points, image)

        points = []

        # print("filtered_points2: ", filtered_points2,"\n\n")
        if (len(filtered_points2) == 4):
            points = filtered_points2
        else:

            # print("Important_points after find_best_corners",important_points)

            # print("\n\n\n\n")

            # 3th approach : removes all the points which are between two others (middle points)
            def dist(p1, p2):
                return int(np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))

            threshold = 1
            points_middle = important_points.copy()
            for i in range(0, len(important_points)):
                A = important_points[i]
                for j in range(0, len(important_points)):
                    B = important_points[j]
                    for k in range(0, len(important_points)):
                        C = important_points[k]
                        if (A[0], A[1]) != (B[0], B[1]) and (A[0], A[1]) != (C[0], C[1]) and (C[0], C[1]) != (
                        B[0], B[1]):
                            if (dist(A, B) + dist(B, C)) - threshold <= dist(A, C) <= (
                                    dist(A, B) + dist(B, C)) + threshold:
                                # print(f"{round(dist(A, C))} = {round(dist(A, B)) + round(dist(B, C))}")
                                # cv2.circle(img, A, 8, (0, 255, 0), -1)
                                if B in points_middle:
                                    points_middle.remove(B)

            # for point in points_middle :
            #   cv.circle(image, point, 7, (255, 0, 255), -1)
            # cv.imshow("Important points", colored_img)
            # cv.waitKey(0)

            # 4th approach : order all points (clockwise) and check distance compared to maximum distance
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

                # Step 1: Calculate pairwise distances between consecutive points (i -> i+1)
                pairwise_distances = []
                for i in range(len(points) - 1):
                    distance = np.linalg.norm(points[i] - points[i + 1])
                    pairwise_distances.append(distance)
                distance = np.linalg.norm(points[len(points) - 1] - points[0])
                pairwise_distances.append(distance)

                # Step 2: Find the maximum distance between consecutive points
                max_distance = np.max(pairwise_distances)

                threshold = max_distance / 1.3

                # Step 3: Visualize threshold
                for i, point in enumerate(points):
                    if i < len(points) - 1:
                        next_point = points[i + 1]
                    else:
                        next_point = points[0]
                    distance = np.linalg.norm(point - next_point)
                    if distance >= threshold:
                        color = (0, 255, 0)  # Green: Points that will be kept
                    else:
                        color = (0, 0, 255)  # Red: Points that will be removed
                    cv.line(image, tuple(point), tuple(next_point), color, 2)

                # Step 4: Filter points: keep only those points whose distance to neighbors is >= threshold
                filtered_points = []
                for i, point in enumerate(points):
                    if i > 0:
                        distance_to_prev = np.linalg.norm(point - points[i - 1])
                    else:
                        distance_to_prev = float('inf')  # First point has no previous neighbor

                    if i < len(points) - 1:
                        distance_to_next = np.linalg.norm(point - points[i + 1])
                    else:
                        distance_to_next = float('inf')  # Last point has no next neighbor

                    # Keep the point only if both distances to neighbors are >= threshold
                    if distance_to_prev >= threshold or distance_to_next >= threshold:
                        filtered_points.append(tuple(point))

                return filtered_points, image

            filtered_points, image_result = remove_close_points(points_middle, colored_img2)

            colored_img = image

            # for point in filtered_points :
            #   cv.circle(colored_img, point, 7, (0, 0, 255), -1)
            # cv.imshow("Important points", image_result)
            # cv.waitKey(0)

            # 5th approach : Keep only the points which could form a rectangle (always 4)
            contour_np = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

            def get_main_corners_from_min_rect(contour, search_radius=20):

                # print("contour de get_main_corners_from_min_rect", contour)

                rect = cv.minAreaRect(contour)  # center, (w, h), angle

                # print("rect", rect)

                box = cv.boxPoints(rect)  # 4 corner points of rotated rectangle

                # print("box", box)

                box = np.intp(box)

                # print("box np", box)

                # Refine: find the closest actual contour point for each box corner
                refined_corners = []
                for corner in box:
                    dists = np.linalg.norm(contour.reshape(-1, 2) - corner, axis=1)
                    closest_pt = contour.reshape(-1, 2)[np.argmin(dists)]
                    refined_corners.append(tuple(closest_pt))

                # print("refined_corners", refined_corners)
                # print("------------------------------------------------------------------------------\n\n")

                return refined_corners

            points = get_main_corners_from_min_rect(contour_np)

            for pt in points:
                cv.circle(image, pt, 10, (255, 255, 0), -1)
            # cv.imshow('Final', colored_img3)
            # cv.waitKey(0)

        def order_corners(corners, cx, cy):
            # Calculate the angle of each corner relative to the centroid
            def calculate_angle(pt):
                dx, dy = pt[0] - cx, pt[1] - cy
                return np.arctan2(dy, dx)

            # Sort corners based on angle
            sorted_corners = sorted(corners, key=calculate_angle)

            # Now, reorder them to ensure top-left, top-right, bottom-right, bottom-left
            top_left = sorted_corners[0]
            bottom_left = sorted_corners[1]
            bottom_right = sorted_corners[2]
            top_right = sorted_corners[3]

            # Now that the points are sorted by angle, we can reorder them to match the required order
            ordered_corners = [top_left, top_right, bottom_right, bottom_left]

            return ordered_corners

        if (points[0] == points[1]) or (points[0] == points[2]) or (points[0] == points[3]) or (points[2] == points[3]):
            print(f"Corners of piece number {piece.index} are incorrect")
            piece.corners = [[-1, -1], [-1, -1], [-1, 1], [1, -1]]
        else:
            # print("Points :",points)
            ordered_points = order_corners(points, cx, cy)
            piece.corners = ordered_points

        # print(f"Corners of piece number {piece.index} are correct",piece.corners)

        #piece.set_picture_debug(colored_img)

        # print("--------------------------------------------------------------------------------------------------\n\n\n")

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Corners saved...")
    return ()
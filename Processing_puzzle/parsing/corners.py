import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def find_corners(myPuzzle, window=False):
    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):
        colored_img = piece.get_color_image().copy()
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

        if window:
            print(f"[{piece.index}] Initial hull points: {len(points)}")
            temp_img = colored_img.copy()
            for pt in points:
                cv.circle(temp_img, pt, 3, (255, 0, 0), -1)
            cv.imshow(f"[{piece.index}] Convex Hull", temp_img)
            cv.waitKey(1)

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
        if window:
            print(f"[{piece.index}] Points through centroid filter: {len(important_points)}")
            temp_img2 = colored_img.copy()
            for pt in important_points:
                cv.circle(temp_img2, pt, 4, (0, 165, 255), -1)
            cv.imshow(f"[{piece.index}] Through Centroid", temp_img2)
            cv.waitKey(1)

        def calculate_centroid(pts):
            return tuple(np.mean(pts, axis=0))

        def calculate_angle(p1, p2, p3):
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi

        def calculate_area(quad):
            x1, y1 = quad[0]
            x2, y2 = quad[1]
            x3, y3 = quad[2]
            x4, y4 = quad[3]
            return 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)

        def calculate_moment_of_inertia(quad, centroid):
            return sum(np.linalg.norm(np.array(p) - np.array(centroid))**2 for p in quad)

        def check_side_lengths(quad):
            lengths = [np.linalg.norm(np.array(quad[i]) - np.array(quad[(i+1)%4])) for i in range(4)]
            return (max(lengths) - min(lengths)) / np.mean(lengths) <= 0.2

        def is_valid_quadrilateral(quad):
            return all(80 <= calculate_angle(quad[i], quad[(i+1)%4], quad[(i+2)%4]) <= 99 for i in range(4))

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
            if best_quads:
                best_quad = best_quads[0][0]
                for pt in best_quad:
                    cv.circle(image, pt, 8, (0, 255, 0), -1)
                return image, list(best_quad)
            return image, []

        image, filtered_points2 = find_best_corners(points, colored_img.copy())
        if window:
            print(f"[{piece.index}] Points after find_best_corners: {len(filtered_points2)}")

        points = []
        if len(filtered_points2) == 4:
            points = filtered_points2
        else:
            if window:
                print(f"[{piece.index}] Fallback: Using intermediate filtering")

            def dist(p1, p2):
                return int(np.linalg.norm(np.array(p1) - np.array(p2)))

            threshold = 1
            points_middle = important_points.copy()
            for i in range(len(important_points)):
                A = important_points[i]
                for j in range(len(important_points)):
                    B = important_points[j]
                    for k in range(len(important_points)):
                        C = important_points[k]
                        if A != B and A != C and C != B:
                            if (dist(A, B) + dist(B, C)) - threshold <= dist(A, C) <= (dist(A, B) + dist(B, C)) + threshold:
                                if B in points_middle:
                                    points_middle.remove(B)

            def order_points_by_angle(all_points):
                centroid = np.mean(all_points, axis=0)
                return sorted(all_points, key=lambda pt: math.atan2(pt[1] - centroid[1], pt[0] - centroid[0]))

            def remove_close_points(all_points, image):
                ordered = order_points_by_angle(all_points)
                points = np.array(ordered)
                pairwise_distances = [np.linalg.norm(points[i] - points[(i+1)%len(points)]) for i in range(len(points))]
                threshold = max(pairwise_distances) / 1.3
                filtered = [tuple(points[i]) for i in range(len(points)) if
                            np.linalg.norm(points[i] - points[(i+1)%len(points)]) >= threshold or
                            np.linalg.norm(points[i] - points[i-1]) >= threshold]
                return filtered, image

            filtered_points, image_result = remove_close_points(points_middle, colored_img2)
            if window:
                print(f"[{piece.index}] Points after remove_close_points: {len(filtered_points)}")
                temp_img3 = colored_img.copy()
                for pt in filtered_points:
                    cv.circle(temp_img3, pt, 5, (0, 0, 255), -1)
                cv.imshow(f"[{piece.index}] Filtered Fallback Points", temp_img3)
                cv.waitKey(1)

            contour_np = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

            def get_main_corners_from_min_rect(contour, search_radius=20):
                box = np.intp(cv.boxPoints(cv.minAreaRect(contour)))
                refined = []
                for corner in box:
                    dists = np.linalg.norm(contour.reshape(-1, 2) - corner, axis=1)
                    closest_pt = contour.reshape(-1, 2)[np.argmin(dists)]
                    refined.append(tuple(closest_pt))
                return refined

            points = get_main_corners_from_min_rect(contour_np)

        def order_corners(corners, cx, cy):
            return sorted(corners, key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))

        if len(points) != 4 or len(set(points)) < 4:
            print(f"Corners of piece number {piece.index} are incorrect")
            piece.corners = [[2, 2], [2, 2], [2, 2], [2, 2]]
        else:
            ordered_points = order_corners(points, cx, cy)
            piece.corners = ordered_points

            if window:
                final_img = colored_img3.copy()
                for pt in ordered_points:
                    cv.circle(final_img, pt, 10, (0, 255, 255), -1)
                cv.imshow(f"Final Corners - Piece {piece.index}", final_img)
                cv.waitKey(0)
                cv.destroyWindow(f"Final Corners - Piece {piece.index}")

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    return ()

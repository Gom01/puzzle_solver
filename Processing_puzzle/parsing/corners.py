import cv2 as cv
import numpy as np
import math
from Processing_puzzle.Puzzle import Puzzle as p

'''
    Function findCorners : Find the 4 corners of a piece [(x,y),(x,y)...]
    Input: myPuzzle
'''

def find_corners(myPuzzle):
    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):

        #Get basics information and conversion
        colored_img = piece.get_color_image()
        colored_img = colored_img.copy()
        colored_img2 = colored_img.copy()
        colored_img3 = colored_img.copy()

        contours = piece.get_contours()
        contour_np = np.array(contours, dtype=np.int32).reshape((-1, 1, 2))

        #Centroid of the piece
        moments = cv.moments(contour_np)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])


        #1st approach : Draw basic polygon arround my piece to get the most important points
        epsilon = 0.005 * cv.arcLength(contour_np, True)
        approx = cv.approxPolyDP(contour_np, epsilon, True)
        hull = cv.convexHull(approx)


        # #Goal always have 4 points
        # cv.polylines(colored_img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        # for pt in hull:
        #      cv.circle(colored_img, tuple(pt[0]), 6, (0, 255, 0), -1)
        # cv.imshow("Important points", colored_img)
        # cv.waitKey(0)


        #2nd approach : using the centroid removes all the points which are not symetric
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
        important_points  = get_points_through_centroid(points, cx, cy, radius=40)

        # for point in important_points :
        #      cv.circle(colored_img, point, 6, (0, 0, 255), -1)
        # cv.imshow("Important points", colored_img)
        # cv.waitKey(0)

        #3th approach : removes all the points which are between two others (middle points)
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
                    if (A[0], A[1]) != (B[0], B[1]) and (A[0], A[1]) != (C[0], C[1]) and (C[0], C[1]) != (B[0], B[1]):
                        if (dist(A, B) + dist(B, C)) - threshold <= dist(A, C) <= (dist(A, B) + dist(B, C)) + threshold:
                            # print(f"{round(dist(A, C))} = {round(dist(A, B)) + round(dist(B, C))}")
                            # cv2.circle(img, A, 8, (0, 255, 0), -1)
                            if B in points_middle:
                                points_middle.remove(B)

        # for point in points_middle :
        #      cv.circle(colored_img, point, 7, (255, 0, 255), -1)
        # cv.imshow("Important points", colored_img)
        # cv.waitKey(0)


        #4th approach : order all points (clockwise) and check distance compared to maximum distance
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

        # for point in filtered_points :
        #     cv.circle(image_result, point, 7, (0, 0, 255), -1)
        # cv.imshow("Important points", image_result)
        # cv.waitKey(0)


        #5th approach : Keep only the points which could form a rectangle (always 4)
        contour_np = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

        def get_main_corners_from_min_rect(contour, search_radius=20):
            rect = cv.minAreaRect(contour)  # center, (w, h), angle
            box = cv.boxPoints(rect)  # 4 corner points of rotated rectangle
            box = np.intp(box)

            # Refine: find the closest actual contour point for each box corner
            refined_corners = []
            for corner in box:
                dists = np.linalg.norm(contour.reshape(-1, 2) - corner, axis=1)
                closest_pt = contour.reshape(-1, 2)[np.argmin(dists)]
                refined_corners.append(tuple(closest_pt))

            return refined_corners


        points = get_main_corners_from_min_rect(contour_np)


        # for pt in points:
        #      cv.circle(colored_img3, pt, 10, (255, 255, 0), -1)
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
            piece.corners = [[-1,-1], [-1,-1], [-1,1], [1,-1]]
        else :
            ordered_points = order_corners(points, cx, cy)
            piece.corners = ordered_points

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Corners saved...")
    return ()

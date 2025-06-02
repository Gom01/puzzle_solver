import cv2
import numpy as np

def straighten_piece(puzzle, window=False, angle_threshold=2.5):
    pieces = puzzle.get_pieces()

    for piece in pieces:
        img = piece.get_color_image()
        corners = piece.get_corners()  # Ordered: 0:TL, 1:BL, 2:BR, 3:TR

        if corners[0] == [2, 2]:
            print(f"⚠️ Skipping piece {piece.index}")
            continue

        # Compute center from moments
        M = piece.get_moment()
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)
        piece.set_position(center)

        def angle_between(p1, p2):
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            return np.degrees(np.arctan2(dy, dx))

        def rotate_image(image, angle, center):
            h, w = image.shape[:2]
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        def rotate_point_around_center(pt, center, angle_rad):
            x, y = pt[0] - center[0], pt[1] - center[1]
            x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            return int(x_new + center[0]), int(y_new + center[1])

        # Compute the angle to rotate so that 0-3 becomes horizontal
        angle = angle_between(corners[0], corners[1])
        apply_rotation = abs(angle) > angle_threshold
        corrected_img = img.copy()

        if apply_rotation:
            corrected_img = rotate_image(img, angle, center)
            piece.set_strait_image(corrected_img)
            sides = piece.get_sides()
            for side in sides:
                side.set_strait_image(corrected_img)


        if window:
            before = img.copy()
            after = corrected_img.copy()
            h, w = before.shape[:2]

            def draw_lines(image, title, angle, corners, center, show_rotated_triangle=False):
                out = image.copy()
                cv2.line(out, (0, center[1]), (w, center[1]), (200, 200, 200), 1)
                cv2.line(out, (center[0], 0), (center[0], h), (200, 200, 200), 1)

                for i, pt in enumerate(corners):
                    cv2.circle(out, pt, 4, (0, 255, 0), -1)
                    cv2.putText(out, f"{i}", (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.line(out, corners[0], corners[3], (255, 0, 0), 2)
                cv2.line(out, corners[0], center, (0, 255, 255), 2)
                cv2.line(out, corners[3], center, (255, 0, 255), 2)
                cv2.circle(out, center, 4, (0, 255, 255), -1)
                cv2.putText(out, f"Angle: {angle:.2f}??", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if show_rotated_triangle:
                    angle_rad = np.radians(-angle)
                    pt0_corr = rotate_point_around_center(corners[0], center, angle_rad)
                    pt3_corr = rotate_point_around_center(corners[3], center, angle_rad)
                    cv2.line(out, pt0_corr, pt3_corr, (0, 255, 255), 2)
                    cv2.line(out, pt0_corr, center, (0, 255, 255), 2)
                    cv2.line(out, pt3_corr, center, (255, 0, 255), 2)
                    cv2.circle(out, pt0_corr, 4, (0, 255, 255), -1)
                    cv2.circle(out, pt3_corr, 4, (255, 0, 255), -1)
                    cv2.putText(out, "Corr0", (pt0_corr[0] - 30, pt0_corr[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.putText(out, "Corr3", (pt3_corr[0] + 5, pt3_corr[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

                    # Compute angle between original and corrected vectors
                    v1 = np.array([corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]])
                    v2 = np.array([pt3_corr[0] - pt0_corr[0], pt3_corr[1] - pt0_corr[1]])
                    dot = np.dot(v1, v2)
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    angle_diff = np.arccos(dot / (norm_v1 * norm_v2))
                    angle_diff_deg = np.degrees(angle_diff)
                    cv2.putText(out, f"Diff: {angle_diff_deg:.2f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                return out

            before_img = draw_lines(before, "Before", angle, corners, center)
            after_img = draw_lines(after, "After", angle, corners, center, show_rotated_triangle=True)

            combined = np.hstack((before_img, after_img))
            cv2.imshow("Before / After with Corrected 0-3 Side", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

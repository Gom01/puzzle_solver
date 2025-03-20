import os
import cv2 as cv2
import numpy as np


'''
    Function find_corners : Find the four corners of a piece, if not found return some close value
    Input: black and white image of a piece
    Output: 4 Corner coordinates [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
'''
def find_corners(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Step 1 : Blur the image to have a better contrast
    blurred_image = cv2.GaussianBlur(gray_img, (9, 9), 0)
    blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)

    #Step 2 : Find all the corners of an image. Should always have the 4 corners
    features = cv2.goodFeaturesToTrack(blurred_image, 15, 0.02, 50, useHarrisDetector=True, k=0.015)
    #print(f"Number of points after goodFeaturesToTrack : {len(corners)}")

    #Step 3 : Remove all the points inside the piece (convex form)
    features = np.int64(features)
    hull = cv2.convexHull(features)
    #print(f"Number of points after convex hull : {len(hull)}")

    #Step 4 : Use the dot product (if close to zero => points are ortho) (all points)
    dots  = {}
    for i in range(0, len(hull)):
        A = hull[i][0]
        for j in range(0, len(hull)):
            B = hull[j][0]
            for k in range(0, len(hull)):
                C = hull[k][0]
                v = C - A
                u = B - A
                #Shouldn't be the same coordinate (==0)
                if (B[0], B[1]) != (A[0], A[1]) and (B[0], B[1]) != (C[0], C[1]) and ((A[0], A[1]) != (C[0], C[1])):
                    dot = abs(np.dot(v,u))
                    if dot not in dots:
                        dots[dot] = {tuple(A), tuple(B), tuple(C)}
                        #print(f"A: {hull[i][0]} B:  {hull[j][0]} C: {hull[k][0]} prod: {dot}")
    #Should use only the best values (closest two zero)
    sorted_dots = {k: dots[k] for k in sorted(dots)}
    points_dot = []
    for index, (key, value) in enumerate(sorted_dots.items()):
        if index < 6:
            value = list(value)
            if value[0] not in points_dot:
                points_dot.append(value[0])
            if value[1] not in points_dot:
                points_dot.append(value[1])
            if value[2] not in points_dot:
                points_dot.append(value[2])
    #print(f"Number of points after dot : {len(points_dot)}")


    #Step 5 : Use distance to remove all the middle points (between two corners)
    def dist(p1, p2):
        return int(np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))
    threshold = 10
    points_middle = points_dot.copy()
    for i in range(0, len(points_dot)):
        A = points_dot[i]
        for j in range(0, len(points_dot)):
            B = points_dot[j]
            for k in range(0, len(points_dot)):
                C = points_dot[k]
                if (A[0], A[1]) != (B[0], B[1]) and (A[0], A[1]) != (C[0], C[1]) and (C[0], C[1]) != (B[0], B[1]):
                    if (dist(A,B) + dist(B,C)) - threshold <= dist(A,C) <= (dist(A,B) + dist(B,C)) + threshold:
                        #print(f"{round(dist(A, C))} = {round(dist(A, B)) + round(dist(B, C))}")
                        #cv2.circle(img, A, 8, (0, 255, 0), -1)
                        if B in points_middle:
                            points_middle.remove(B)
    #print(f"Number of points removing middle points: {len(new_corners)}")


    #Step 6 : Get final corners based on point missing (calculate missing coordinates)
    def calculate_missing(points_middle):
        threshold = 45
        for i in range(0, len(points_middle)):
            A = points_middle[i]
            for j in range(0, len(points_middle)):
                B = points_middle[j]
                for k in range(0, len(points_middle)):
                    C = points_middle[k]
                    if (A[0], A[1]) != (B[0], B[1]) and (A[0], A[1]) != (C[0], C[1]) and (C[0], C[1]) != (B[0], B[1]):
                        Ax,Ay = A
                        Bx, By = B
                        Cx, Cy = C
                        x_length = Bx + Cx - Ax
                        y_length = By + Cy - Ay

                        for c in points_middle:
                            if x_length - threshold <=  c[0] <= x_length + threshold:
                                if y_length - threshold <= c[1] <= y_length + threshold:
                                    finalCorners = [A,B,C,c]
                                    return(finalCorners)
        if len(points_middle) >= 4:
            return [points_middle[0], points_middle[1], points_middle[2], points_middle[3]]
        else:
            return [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]

    return calculate_missing(points_middle)


# ##Main code :
#
# path = '../Shape_Detector/list_pieces'
#
# for filename in os.listdir(path):
#     img_path = os.path.join(path, filename)
#     img = cv2.imread(img_path)
#
#     corners = find_corners(img)
#     for c in corners:
#         cv2.circle(img, c, 5, (255, 0, 0), -1)
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
























    # ## ******* Finding center of a piece *****
    #
    # #Using distance transform (the furthest from black the whiter it is => find center of piece)
    # image = cv2.imread(img_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # plt.imshow(image, cmap='gray')
    # plt.show()
    #
    # # Create a binary image by throttling the image.
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # # Determine the distance transform.
    # dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    #
    # # Make the distance transform normal.
    # dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    #
    # _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist, None)
    #
    #
    # # plt.imshow(image)
    # # plt.scatter(maxLoc[0], maxLoc[1])
    # # plt.show()
    # # plt.imshow(dist_output)
    # # plt.scatter(maxLoc[0], maxLoc[1])
    # # plt.show()
    #
    #
    # ## ******* Finding contour of a piece ***** (Need to use Florian's algorithm)
    #
    # edged = cv2.Canny(image, 50, 150)
    # contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # #plt.imshow(image)
    # #plt.show()
    #
    #
    # ##TODOO
    # for count in contours:
    #     epsilon = 0.01 * cv2.arcLength(count, True)
    #     approximations = cv2.approxPolyDP(contour, epsilon, True)
    #     cv2.drawContours(gray, [approximations], 0, (0), 3)
    #
    # ## ******* Creating all vectors from center to contour *****
    #
    # center = [maxLoc[0], maxLoc[1]]
    # distances = []
    # points = []
    # for contour in contours:
    #     for point in contour:
    #         #plt.imshow(image)
    #         #plt.plot([center[0], point[0][0]], [center[1], point[0][1]], linestyle='-', marker='o', color='b')
    #         #plt.show()
    #         points.append(point[0])
    #         distances.append(np.linalg.norm(np.array(center) - np.array(point[0])))
    #
    #
    # peaks, _ = find_peaks(distances, prominence=5)
    #
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    #
    # important_points = []
    #
    # # First subplot: Plot the distances and peaks
    # axs[0].plot(distances)
    # for peak in peaks:
    #     axs[0].plot(peak, distances[peak], "x")
    # axs[0].set_title("Distances with Peaks")
    # axs[0].set_xlabel("Index")
    # axs[0].set_ylabel("Distance")
    # axs[0].grid(True)
    #
    # # Second subplot: Plot the image and points with peaks marked
    # axs[1].imshow(image)
    # axs[1].plot(center[0], center[1], "go")  # Plot the center as a green circle
    # for peak in peaks:
    #     important_points.append([int(points[peak][0]), int(points[peak][1])])
    #     axs[1].plot([center[0], points[peak][0]], [center[1], points[peak][1]], linestyle='-', marker='o', color='b')
    #     axs[1].plot(points[peak][0], points[peak][1], "rx")  # Plot the peaks as red 'x' markers
    # axs[1].set_title("Image with Peaks")
    # axs[1].axis('off')  # Hide axis for the image plot
    #
    # print(important_points)
    #
    # # Show the combined plot
    # plt.tight_layout()  # Adjust layout for better spacing
    # plt.show()
    # plt.close()


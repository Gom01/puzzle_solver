import cv2 as cv2
import numpy as np


'''
    Function raw_corners : Find the four corners of a piece, if not found return some close value these points
    doesn't correspond to real points
    Input: black and white image of a piece
    Output: 4 Corner coordinates [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
'''
def raw_corners(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Step 1 : Blur the image to have a better contrast
    blurred_image = cv2.GaussianBlur(gray_img, (9, 9), 0)
    blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)

    #Step 2 : Find all the corners of an image. Should always have the 4 corners
    features = cv2.goodFeaturesToTrack(blurred_image, 15, 0.02, 50, useHarrisDetector=True, k=0.015)
    #print(f"Number of points after goodFeaturesToTrack : {len(features)}")
    #Step 3 : Remove all the points inside the piece (convex form)
    if features is None:
        print("Something wierd happened (black image !?)")
        return [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
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


'''
    Function find_corners : Match raw_corners to the value
'''
def find_corners(myPuzzle):
        pieces = myPuzzle.get_pieces()

        for idx, piece in enumerate(pieces):
            img = piece.get_black_white_image()
            contours = piece.get_contours()
            rawCorners = raw_corners(img)

            dist_mini_corners_1 = float('inf')
            dist_mini_corners_2 = float('inf')
            dist_mini_corners_3 = float('inf')
            dist_mini_corners_4 = float('inf')

            corner_adjusted_1 = rawCorners[0]
            corner_adjusted_2 = rawCorners[1]
            corner_adjusted_3 = rawCorners[2]
            corner_adjusted_4 = rawCorners[3]

            x0, y0 = rawCorners[0]
            x1, y1 = rawCorners[1]
            x2, y2 = rawCorners[2]
            x3, y3 = rawCorners[3]

            if rawCorners[0] != [-1,-1]:
                for point in contours:
                    x, y = point[0], point[1]

                    dist_corner_1 = ((x - x0)**2.0 + (y - y0)**2.0)**(1.0/2.0)
                    dist_corner_2 = ((x - x1)**2.0 + (y - y1)**2.0)**(1.0 / 2.0)
                    dist_corner_3 = ((x - x2)**2.0 + (y - y2)**2.0)**(1.0 / 2.0)
                    dist_corner_4 = ((x - x3)**2.0 + (y - y3)**2.0)**(1.0 / 2.0)

                    if dist_corner_1 < dist_mini_corners_1:
                         dist_mini_corners_1 = dist_corner_1
                         corner_adjusted_1 = (x,y)

                    if dist_corner_2 < dist_mini_corners_2:
                           dist_mini_corners_2 = dist_corner_2
                           corner_adjusted_2 = (x,y)

                    if dist_corner_3 < dist_mini_corners_3:
                        dist_mini_corners_3 = dist_corner_3
                        corner_adjusted_3 = (x,y)

                    if dist_corner_4 < dist_mini_corners_4:
                        dist_mini_corners_4 = dist_corner_4
                        corner_adjusted_4 = (x,y)


                piece.set_corners([corner_adjusted_1,corner_adjusted_2,corner_adjusted_3,corner_adjusted_4])
            else:
                piece.set_corners([[-1,-1],[-1,-1],[-1,-1], [-1,-1]])

        myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
        print("Corners saved ! ")
        return()

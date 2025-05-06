from Processing_puzzle import Puzzle as p
import numpy as np
import cv2

'''find_color() find the colors of all the pixel which are inside the contour.
    input : puzzle and possible to modify the factor(should be very small) 
    color_contour is an array of color [(b,g,r),(b,g,r)...] '''
def find_color(puzzle, factor=0):
    pieces = puzzle.get_pieces()
    for idx, piece in enumerate(pieces):
        contours = piece.get_contours()
        contour_color = []

        #Find center of the piece for optimizing color
        cx, cy = -1,-1
        M = cv2.moments(np.array(contours))
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else :
            print("Center not found ! ")
        piece.x, piece.y = cx,cy

        #Creating new mask image
        black_mask = np.zeros(piece.image_color.shape[:2], dtype=np.uint8)
        for point in contours:
            cv2.circle(black_mask, (point[0], point[1]), 10, (255, 255, 255), -1)

        #Change white pixels to colored pixel
        applied_mask_image = cv2.bitwise_and(piece.image_color, piece.image_color, mask=black_mask)
        #cv2.imshow('Contours', applied_mask_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        ##Going through the contours take the color which is more inside (factor)
        for point in contours:
            x,y = point

            move_x = int((cx - x) * factor)
            move_y = int((cy - y) * factor)
            new_x = x + move_x
            new_y = y + move_y

            img = piece.image_black_white
            color_point = applied_mask_image[new_y, new_x]
            b, g, r = color_point
            contour_color.append([b,g,r])
            #cv2.circle(img, point, 7, (int(b),int(g),int(r)), -1)

        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        piece.colors_contour = contour_color

    puzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("Colored contour saved ! ")
    return()


#myPuzzle = p.Puzzle()
#myPuzzle.load_puzzle('../Processing_puzzle/res/puzzle.pickle')
#pieces = myPuzzle.get_pieces()
#find_color(myPuzzle)

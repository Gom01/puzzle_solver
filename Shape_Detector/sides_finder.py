import cv2

from Puzzle import Puzzle
from Shape_Detector.corner_finder import find_corners

# Créer une instance de Puzzle
puzzle = Puzzle()

# Charger le puzzle depuis le fichier pickle
puzzle.load_puzzle('mon_puzzle.pickle')

# Vérifier que les pièces sont chargées correctement
pieces = puzzle.get_pieces()

# Afficher ou traiter les pièces du puzzle
for idx, piece in enumerate(pieces):
    # Récupérer l'image couleur
    img = piece.get_image()

    cv2.imshow('image', img)

    img_color = piece.get_image_color()

    contours = piece.get_contours()

    #print("Contours : ",contours)

    #cv2.drawContours(img_color, contours,-1, (0, 255, 255), 2,maxLevel=1)

    corners = find_corners(img)

    #print("Corners : ",corners)

    dist_mini_corners_1 = float('inf')
    dist_mini_corners_2 = float('inf')
    dist_mini_corners_3 = float('inf')
    dist_mini_corners_4 = float('inf')

    corner_adjusted_1 = corners[0]
    corner_adjusted_2 = corners[1]
    corner_adjusted_3 = corners[2]
    corner_adjusted_4 = corners[3]

    x0, y0 = corners[0]
    x1, y1 = corners[1]
    x2, y2 = corners[2]
    x3, y3 = corners[3]

    #print("Corners : ",corners[0])

    if not(corners[0] == corners[1] == corners[2] == corners[3]):

        for contour in contours:
            for point in contour:
                x, y = point[0]



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




        #print("Corners_Adjusted : ",corner_adjusted_1," - ",corner_adjusted_2," - ",corner_adjusted_3," - ",corner_adjusted_4)

        corners_adjusted = [corner_adjusted_1,corner_adjusted_2,corner_adjusted_3,corner_adjusted_4]

    else :
        corners_adjusted = []
        #print(len(corners_adjusted))

    piece.find_4_sides(corners_adjusted)

    side1, side2, side3, side4 = piece.get_4_sides()

    # print("side1 : ",side1)

    img_color_S = piece.display_4_sides(img_color, False, 0)

    for c in corners:
        cv2.circle(img_color_S, c, 5, (255, 0, 0), -1)

    for c in corners_adjusted:
        cv2.circle(img_color_S, c, 5, (255,0,255), -1)

    cv2.imshow("img", img_color_S)
    cv2.waitKey(1000)







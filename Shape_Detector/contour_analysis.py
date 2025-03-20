import corner_finder as cf
import cv2

path = "../Shape_Detector/list_pieces/piece28_9.jpg"

img = cv2.imread(path)
points = cf.find_corners(img)

print(points)
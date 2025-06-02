import cv2
import numpy as np

# Charger l'image
image_path = '../images/p1_b/Natel.Black1.jpg'  # Vérifiez le chemin de l'image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"L'image '{image_path}' n'a pas été trouvée.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un flou pour réduire le bruit du fond
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Appliquer un seuillage adaptatif pour isoler les pièces
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Fermer les petites zones bruitées
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Appliquer la distance transform (version OpenCV)
distance_map = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# Normaliser et convertir en uint8
_, distance_map = cv2.threshold(distance_map, 0.7 * distance_map.max(), 255, cv2.THRESH_BINARY)
distance_map = np.uint8(distance_map)

# Trouver les marqueurs avec connectedComponents (remplace watershed de skimage)
num_labels, markers = cv2.connectedComponents(distance_map)

# Appliquer Watershed d'OpenCV
markers = markers + 1
markers[thresh == 0] = 0
cv2.watershed(image, markers)

# Initialiser la surface totale des objets détectés
total_area = 0

# Détecter les contours à partir des résultats de Watershed
for label in range(2, num_labels + 1):  # Ignore le fond (0) et le bord (1)
    mask = np.uint8(markers == label) * 255

    # Trouver les contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        continue

    # Récupérer le plus grand contour
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area += area

    # Dessiner le contour sur l'image
    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)

print(f"Surface totale des pièces détectées: {total_area} pixels²")

# Afficher l'image avec les contours détectés
cv2.imshow('Pièces détectées', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

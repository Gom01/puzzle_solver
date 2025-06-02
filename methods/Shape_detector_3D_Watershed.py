import cv2
import numpy as np
import os

WINDOW_SIZE = (1000, 800)

# Dossier contenant les images
dossier_images = "../images/p3"

# Vérification de l'existence du dossier
if not os.path.exists(dossier_images):
    raise ValueError(f"Le dossier {dossier_images} n'existe pas.")

# Liste des fichiers image dans le dossier
fichiers = sorted([f for f in os.listdir(dossier_images) if f.endswith(('.jpg', '.png', '.jpeg'))])

# Vérification s'il y a suffisamment d'images
if len(fichiers) < 2:
    raise ValueError("Il faut au moins 2 images pour l'alignement et la fusion !")

# Charger la première image comme référence
ref_image = cv2.imread(os.path.join(dossier_images, fichiers[0]), cv2.IMREAD_GRAYSCALE)

# Détecteur de points clés ORB
orb = cv2.ORB_create()

# Liste pour stocker les images alignées
images_alignees = [ref_image]

# Aligner toutes les images sur la première
for fichier in fichiers[1:]:
    img = cv2.imread(os.path.join(dossier_images, fichier), cv2.IMREAD_GRAYSCALE)

    # Détection des points clés et descripteurs
    kp1, des1 = orb.detectAndCompute(ref_image, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    # Vérifier qu'on a trouvé des points
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        print(f"⚠️ Pas assez de points détectés pour {fichier}, on l'ignore.")
        continue

    # Appariement des points
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Vérifier qu'on a assez de correspondances
    if len(matches) < 5:
        print(f"⚠️ Pas assez de correspondances entre {fichiers[0]} et {fichier}. Ignoré.")
        continue

    # Extraction des points correspondants
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calcul de l'homographie et transformation
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = ref_image.shape
    img_aligned = cv2.warpPerspective(img, H, (w, h))

    images_alignees.append(img_aligned)

# Fusion des images alignées en utilisant la médiane (plus robuste que la moyenne)
fused_image = np.median(np.array(images_alignees), axis=0).astype(np.uint8)

# Affichage de l'image fusionnée
cv2.imshow("Image Fusionnée", cv2.resize(fused_image,WINDOW_SIZE))
cv2.waitKey(0)

# **Segmentation pour extraire les pièces**
# Seuillage adaptatif pour une meilleure détection
thresh = cv2.adaptiveThreshold(fused_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Morphologie pour nettoyer les bruits
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Détection des contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Création d'un masque pour les pièces
mask = np.zeros_like(fused_image)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Appliquer Watershed pour améliorer la séparation
dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
_, markers = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
markers = np.int32(markers)
markers = cv2.watershed(cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR), markers)

# Corriger les frontières mal détectées
mask[markers == -1] = 0  # Supprimer les frontières du Watershed

# Affichage des résultats finaux
cv2.imshow("Contours détectés", cv2.resize(mask,WINDOW_SIZE))
cv2.waitKey(0)
cv2.destroyAllWindows()

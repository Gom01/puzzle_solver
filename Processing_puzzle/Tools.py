import cv2 as cv
import numpy as np

#Basic settings
WINDOW_SIZE = (1000, 800)
WINDOW_SIZE_PIECE = (200, 200)
X_PERCENT = 50  # Ajustable (ex : 30% autour de la médiane)
MARGIN_PERCENT = 1  # Ajuste la taille de la marge autour des contours

class Tools:
    """
    Cette classe contient des méthodes utilitaires pour le traitement d'image avec OpenCV.
    Chaque méthode peut afficher l'image résultante selon les besoins.
    """
    def convert_to_grayscale(self, image, window=False, window_size=None, time=0):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Grayscale', cv.resize(gray, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return gray

    def apply_otsu_threshold(self, image, window=False, window_size=None, time=0):
        _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Otsu Threshold', cv.resize(thresh, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return thresh

    def remove_noise(self, image, kernel_size, iterations, window=False, window_size=None, time=0):
        kernel = np.ones(kernel_size, np.uint8)
        opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)
        opening = cv.bitwise_not(opening)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Noise Removal', cv.resize(opening, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return opening

    def find_contours(self,image_origin, image, window=False, window_size=None, time=0):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if window:
            image_c = cv.drawContours(image_origin.copy(), contours, -1, (0, 255, 0), 2)
            cv.imshow('Contours', cv.resize(image_c, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return contours,_

    def filter_contours_by_area(self,image_origin, contours, min_area, window=False, window_size=None, time=0):
        areas = [cv.contourArea(cnt) for cnt in contours]
        areas_filtered = [area for area in areas if area > min_area]

        # Calcul de la médiane et définition des bornes selon X%
        median_area = np.median(areas_filtered) if areas_filtered else 0
        min_valid_area = median_area * (1 - X_PERCENT / 100)
        max_valid_area = median_area * (1 + X_PERCENT / 100)

        # Filtrer les contours basés sur la médiane
        contours_filtered = [cnt for cnt in contours if min_valid_area <= cv.contourArea(cnt) <= max_valid_area]


        mask = np.zeros(image_origin.shape[:2], dtype=np.uint8)

        # Dessiner les contours sur le masque
        cv.drawContours(mask, contours_filtered, -1, (255), thickness=cv.FILLED)

        # Appliquer une dilatation pour créer une marge autour des contours
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv.dilate(mask, kernel, iterations=MARGIN_PERCENT)

        # Inverser le masque dilaté pour avoir les zones extérieures en noir
        mask_inv = cv.bitwise_not(mask_dilated)

        # Appliquer le masque inversé pour transformer l'extérieur en noir
        im_with_black_background = cv.bitwise_and(image_origin, image_origin, mask=mask_dilated)

        if window:
            print(f"Aire médiane : {median_area}")
            print(f"Intervalle de sélection : [{min_valid_area}, {max_valid_area}]")

            # Afficher l'image avec le fond noir
            cv.imshow('Image avec arrière-plan noir', cv.resize(im_with_black_background, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return im_with_black_background,contours_filtered,min_valid_area,max_valid_area



    def get_bounding_rect(self,image, contour, margin=10, window=False, window_size=None, time=0):
        x, y, w, h = cv.boundingRect(contour)
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w += 2 * margin
        h += 2 * margin

        # Assurer que les coordonnées du rectangle restent positives
        x = max(x, 0)
        y = max(y, 0)

        piece_image = image[y:y + h, x:x + w]

        if window:

            cv.imshow('Bounding Rect', cv.resize(piece_image.copy(), window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return x, y, w, h, piece_image

    def save_image(self, image, path, window=False, window_size=None, time=0):
        cv.imwrite(path, image)
        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow(f'Image saved at {path}', cv.resize(image, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()

    def apply_sobel(self, image, window=False, window_size=None, time=0):
        sobelx_piece = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
        sobely_piece = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
        sobel_piece = cv.magnitude(sobelx_piece, sobely_piece)
        sobel_piece = cv.convertScaleAbs(sobel_piece)

        if window:
            window_size = window_size or WINDOW_SIZE
            cv.imshow('Sobel', cv.resize(sobel_piece, window_size))
            cv.waitKey(time)
            cv.destroyAllWindows()
        return sobel_piece


import cv2 as cv
import numpy as np

class Piece:
    index = 0
    def __init__(self, image_black_white,image_color, contours, x, y):
        """
        Initialise une pièce de puzzle.
        :param image_black_white: L'image de la pièce avec le fond noir et l'intérieur blanc.
        :param contours: Les contours filtrés de la pièce.
        :param x: Coordonnée x de la pièce.
        :param y: Coordonnée y de la pièce.
        """
        self.image_black_white = image_black_white
        self.image_color = image_color
        self.contours = contours
        self.adjusted_contours = None
        self.x = x
        self.y = y
        self.corners = None
        self.colors_contour = []
        self.moments = None
        self.side_1 = None
        self.side_2 = None
        self.side_3 = None
        self.side_4 = None
        self.index += 1


    def get_black_white_image(self):
        return self.image_black_white

    def get_color_image(self):
        return self.image_color

    def get_corners(self):
        return self.corners

    def get_contours(self):
        adjusted_contours = []

        for contour in self.contours:
            adjusted_contour = contour - np.array([[self.x, self.y]])
            adjusted_contours.append(adjusted_contour)

        self.adjusted_contours = adjusted_contours

        return adjusted_contours


    def get_position(self):
            """Retourne la position (x, y) de la pièce."""
            return self.x, self.y

    def find_color_contour(self, shift_factor=3):
        """
        Trouve les couleurs sur le contour et les décale vers l'intérieur de la pièce.

        :param shift_factor: Distance du déplacement des pixels vers le centre.
        """
        self.colors_contour = []

        # Créer un masque noir de la même taille que l'image
        mask = np.zeros(self.image_black_white.shape[:2], dtype=np.uint8)
        contours_filtered_p = self.get_contours()

        # Dessiner les contours filtrés sur le masque
        #cv.drawContours(mask, contours_filtered_p, -1, (255), thickness=4)

        # Extraire les pixels qui appartiennent au contour
        contour_pixels = cv.bitwise_and(self.image_color, self.image_color, mask=mask)

        # Trouver le centre de la pièce en utilisant les moments
        M = cv.moments(mask)

        self.moments = M

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            self.moments = [cX, cY]
        else:
            cX, cY = self.x, self.y  # Si le moment est invalide, on garde la position initiale

        # Itérer sur les pixels du contour
        for y in range(contour_pixels.shape[0]):
            for x in range(contour_pixels.shape[1]):
                if mask[y, x] == 255:  # Vérifier si le pixel fait partie du contour
                    color = tuple(contour_pixels[y, x])

                    # Calcul du vecteur directionnel vers le centre
                    dir_x = cX - x
                    dir_y = cY - y
                    norm = np.sqrt(dir_x ** 2 + dir_y ** 2)

                    if norm != 0:  # Éviter la division par zéro
                        dir_x = int(shift_factor * dir_x / norm)
                        dir_y = int(shift_factor * dir_y / norm)

                    # Nouvelle position après déplacement vers l'intérieur
                    new_x = min(max(x + dir_x, 0), self.image_black_white.shape[1] - 1)
                    new_y = min(max(y + dir_y, 0), self.image_black_white.shape[0] - 1)

                    # Ajouter la couleur du pixel déplacé
                    self.colors_contour.append({'x': new_x, 'y': new_y, 'color': color})

        # Affichage des 10 premiers résultats pour vérification
        print("Exemples de couleurs des contours après déplacement :", self.colors_contour[:10])



    def get_color_contour(self):
        return self.colors_contour


    def get_moments(self):

        self.find_color_contour()

        return self.moments

    def display_color_pixels(self):
        """Affiche chaque pixel de contour en grand, un par un."""
        for data in self.colors_contour:
            color = np.full((1000, 1000, 3), data['color'], dtype=np.uint8)

            cv.imshow(f"Pixel ({data['x']}, {data['y']})", color)
            key = cv.waitKey(1000)  # Attendre une touche pour passer au suivant
            if key == 27:  # Si Échap est pressé, sortir
                break
            cv.destroyAllWindows()  # Fermer avant d'afficher le suivant

        cv.destroyAllWindows()

    def get_next_point(self):
        """
        Récupère le point suivant dans le contour (parcours circulaire).

        :return: Tuple (x, y) du point suivant.
        """
        if not hasattr(self, 'index'):  # Vérifie si l'attribut 'index' existe
            self.index = 0  # Initialisation de l'attribut 'index' si nécessaire

        if len(self.get_contours()) == 0:  # Vérifie si les contours sont vides
            print("Erreur: Aucun contour trouvé.")
            return None

        contour = self.get_contours()[0]  # On suppose qu'il y a au moins un contour
        if len(contour) == 0:  # Vérifie si le contour est vide
            print("Erreur: Le contour est vide.")
            return None

        # Assure-toi que l'index reste dans la plage du contour
        self.index = (self.index + 1) % len(contour)

        return tuple(contour[self.index][0])  # Retourne (x, y)

    def find_4_sides(self, list_corners):

        if (len(list_corners) == 4):

            self.list_corners = list_corners

            corner_1 = list_corners[0]
            corner_2 = list_corners[1]
            corner_3 = list_corners[2]
            corner_4 = list_corners[3]

            # Initialisation des côtés
            side1 = []
            side2 = []
            side3 = []
            side4 = []

            # Définition de l'ordre des coins, pour éviter toute confusion lors de l'itération
            corners = [corner_1, corner_2, corner_3, corner_4]

            # Point initial du contour
            point_origin = self.get_next_point()

            while point_origin != corner_1:
                point_origin = self.get_next_point()

            # print(point_origin," - ",corner_1)

            point_origin = self.get_next_point()

            # print(point_origin," - ",corner_1)

            point = point_origin

            # Variable pour suivre quel côté nous sommes en train de remplir
            current_side = side1
            current_corner_index = 0

            # Liste des coins que nous avons rencontrés pour la première fois
            visited_corners = []
            visited_corners.append(corner_1)
            corners.remove(corner_1)

            # Tant qu'on n'a pas parcouru tout le contour
            while point != point_origin or len(visited_corners) < 4:
                # Récupérer le prochain point sur le contour
                point = self.get_next_point()

                # Vérifie si le point est un coin
                if point in corners:
                    # Si nous passons d'un coin à un autre, on commence un nouveau côté
                    if point not in visited_corners:
                        visited_corners.append(point)
                        current_corner_index += 1

                        # On change de côté chaque fois qu'on passe à un coin
                        if current_corner_index == 1:
                            current_side = side2
                        elif current_corner_index == 2:
                            current_side = side3
                        elif current_corner_index == 3:
                            current_side = side4

                # Ajoute le point au côté actuel
                current_side.append(point)

                self.side_1 = side1
                self.side_2 = side2
                self.side_3 = side3
                self.side_4 = side4

    def get_next_point(self):
        """
        Récupère le point suivant dans le contour (parcours circulaire).

        :return: Tuple (x, y) du point suivant.
        """
        if not hasattr(self, 'index'):  # Vérifie si l'attribut 'index' existe
            self.index = 0  # Initialisation de l'attribut 'index' si nécessaire

        if len(self.get_contours()) == 0:  # Vérifie si les contours sont vides
            print("Erreur: Aucun contour trouvé.")
            return None

        contour = self.get_contours()[0]  # On suppose qu'il y a au moins un contour
        if len(contour) == 0:  # Vérifie si le contour est vide
            print("Erreur: Le contour est vide.")
            return None

        # Assure-toi que l'index reste dans la plage du contour
        self.index = (self.index + 1) % len(contour)

        return tuple(contour[self.index][0])  # Retourne (x, y)

    def get_4_sides(self):
        if not (self.side_1 is None or self.side_2 is None or self.side_3 is None or self.side_4 is None):
            # Si tous les côtés sont définis, on peut procéder à une action

            self.find_4_sides(self.list_corners)

        return self.side_1, self.side_2, self.side_3, self.side_4

    def display_4_sides(self, img, window=False, time=0):

        self.get_4_sides()

        if not (self.side_1 is None or self.side_2 is None or self.side_3 is None or self.side_4 is None):
            # Si tous les côtés sont définis, on peut procéder à une action
            # print("Tous les côtés sont définis. On peut continuer.")

            color_side1 = (255, 0, 0)  # Bleu
            color_side2 = (0, 255, 0)  # Vert
            color_side3 = (0, 0, 255)  # Rouge
            color_side4 = (255, 255, 0)  # Jaune

            # Dessiner la ligne reliant les points
            for i in range(len(self.side_1) - 1):
                cv.line(img, self.side_1[i], self.side_1[i + 1], color_side1, 4)

            # Dessiner la ligne reliant les points
            for i in range(len(self.side_2) - 1):
                cv.line(img, self.side_2[i], self.side_2[i + 1], color_side2, 4)

            # Dessiner la ligne reliant les points
            for i in range(len(self.side_3) - 1):
                cv.line(img, self.side_3[i], self.side_3[i + 1], color_side3, 4)

            # Dessiner la ligne reliant les points
            for i in range(len(self.side_4) - 1):
                cv.line(img, self.side_4[i], self.side_4[i + 1], color_side4, 4)

            if window:
                cv.imshow("4 Sides", img)
                cv.waitKey(time)
                cv.destroyAllWindows()
            return img





        else:
            # Si l'un des côtés est manquant, on peut soit demander à les définir, soit sortir
            print("Il manque des côtés. Veuillez vérifier.")
            return img
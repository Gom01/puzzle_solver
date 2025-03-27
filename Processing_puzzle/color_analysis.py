## TODO Correct this part (add directly to the contour)




 ##TODO mettre dans un autre dossier (fichier.py)
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

 ## TODO put in the same file as find_color function and direct apply transformation
    def get_moments(self):
        #Shouldn't have function for get()
        self.find_color_contour()
        return self.moments



    # def display_color_pixels(self):
    #     """Affiche chaque pixel de contour en grand, un par un."""
    #     for data in self.colors_contour:
    #         color = np.full((1000, 1000, 3), data['color'], dtype=np.uint8)
    #
    #         cv.imshow(f"Pixel ({data['x']}, {data['y']})", color)
    #         key = cv.waitKey(1000)  # Attendre une touche pour passer au suivant
    #         if key == 27:  # Si Échap est pressé, sortir
    #             break
    #         cv.destroyAllWindows()  # Fermer avant d'afficher le suivant
    #
    #     cv.destroyAllWindows()


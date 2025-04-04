import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, splprep, splev

from Processing_puzzle import Puzzle as p

def sides_eq(myPuzzle):

    def drawing_sides(piece, windows=False):

        # Centre de masse du contour global de la pièce
        piece_contour_R = np.array(piece.get_contours()).reshape((-1, 1, 2))
        M1 = cv.moments(piece_contour_R)

        sides = piece.get_sides()


        types = [sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info()]

        list_eq = []

        for i in range(4):

            type = types[i]

            points = sides[i].get_side_points()

            points = np.array(points, dtype=np.int32).reshape((-1, 2))


            if windows:
                picture = piece.get_black_white_image()
                cv2.drawContours(picture, [points], 0, (255, 0, 255), 2)
                cv2.imshow('picture', picture)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            cx = int(M1["m10"] / M1["m00"])
            cy = int(M1["m01"] / M1["m00"])

            # Étape 1 : Translation pour ramener le premier point à (0,0)
            x0, y0 = points[0]
            translated_points = [(x - x0, y - y0) for x, y in points]

            # Nouveau dernier point après translation
            x1_t, y1_t = translated_points[-1]

            # Étape 2 : Calcul de l'angle de rotation
            theta = np.arctan2(y1_t, x1_t)  # Angle entre le premier et dernier point

            # Matrice de rotation
            cos_t, sin_t = np.cos(-theta), np.sin(-theta)
            rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            # Application de la rotation
            rotated_points = [tuple(np.dot(rotation_matrix, [x, y])) for x, y in translated_points]

            # Transformation des coordonnées du centre de masse (cx, cy)
            transformed_cx, transformed_cy = np.dot(rotation_matrix, [cx - x0, cy - y0])

            # Si la pièce est concave, appliquer une rotation finale de 180 degrés
            if type == "concave":
                # Matrice de rotation de 180 degrés
                rotation_180 = np.array([[-1, 0], [0, -1]])

                # Appliquer la rotation de 180 degrés aux points transformés
                rotated_points = [tuple(np.dot(rotation_180, [x, y])) for x, y in rotated_points]

                # Transformation du centre de masse
                transformed_cx, transformed_cy = np.dot(rotation_180, [transformed_cx, transformed_cy])

            # Séparation des coordonnées X et Y
            x_values, y_values = zip(*rotated_points)


            if windows:
                # Création de la courbe
                plt.figure(figsize=(8, 5))
                plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5, label="Points transformés")

                # Ajout du centre de masse transformé
                plt.scatter(transformed_cx, transformed_cy, color='r', label=f'Centre de masse ({transformed_cx:.2f}, {transformed_cy:.2f})', zorder=5)

                plt.axhline(0, color='black', linewidth=1)  # Ligne horizontale
                plt.axvline(0, color='black', linewidth=1)  # Ligne verticale
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.title("Courbe avec points alignés et centre de masse")
                plt.xlabel("X aligné")
                plt.ylabel("Y ajusté")
                plt.legend()

                # Affichage
                plt.show()

            s = 150

            t_values = np.linspace(0, 1, len(x_values))  # Paramètre t sur l'intervalle [0, 1]
            parametric_curve = np.array([(x_values[i], y_values[i]) for i in range(len(x_values))])

            tck1, u1 = splprep([x_values, y_values], s=s, k=2)

            tck2, u2 = splprep([x_values, y_values], s=s, k=3)

            tck3, u3 = splprep([x_values, y_values], s=s, k=5)


            if windows:

                # --- Configuration du graphique ---
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()

                # --- Paramétrique (de base) ---


                axes[0].plot(t_values, parametric_curve[:, 0], 'bo-', label="X (paramétrique)")
                axes[0].plot(t_values, parametric_curve[:, 1], 'go-', label="Y (paramétrique)")
                axes[0].legend()
                axes[0].grid(True)
                axes[0].set_title("Courbe paramétrique des points")
                axes[0].set_xlabel("Paramètre t")
                axes[0].set_ylabel("Valeurs de X et Y")



                # --- Spline paramétrique

                new_points1 = splev(np.linspace(0, 1, 100), tck1)
                axes[1].plot(x_values, y_values, 'bo', label="Points originaux")
                axes[1].plot(new_points1[0], new_points1[1], 'r-', label="")
                axes[1].legend()
                axes[1].grid(True)
                axes[1].set_title(f"Spline paramétrique (s=${s}, k=2)")
                axes[1].set_xlabel("X")
                axes[1].set_ylabel("Y")

                # --- Spline paramétrique

                new_points2 = splev(np.linspace(0, 1, 100), tck2)
                axes[2].plot(x_values, y_values, 'bo', label="Points originaux")
                axes[2].plot(new_points2[0], new_points2[1], 'r-', label="")
                axes[2].legend()
                axes[2].grid(True)
                axes[2].set_title(f"Spline paramétrique (s=${s}, k=3)")
                axes[2].set_xlabel("X")
                axes[2].set_ylabel("Y")

                # --- Spline paramétrique

                new_points3 = splev(np.linspace(0, 1, 100), tck3)
                axes[3].plot(x_values, y_values, 'bo', label="Points originaux")
                axes[3].plot(new_points3[0], new_points3[1], 'r-', label="")
                axes[3].legend()
                axes[3].grid(True)
                axes[3].set_title(f"Spline paramétrique (s=${s}, k=5)")
                axes[3].set_xlabel("X")
                axes[3].set_ylabel("Y")

                # Affichage final des graphes
                plt.tight_layout()
                plt.show()


            sides[i].set_side_eq((tck2,u2))

            list_eq.append((tck2,u2))

        return list_eq




    pieces = myPuzzle.get_pieces()

    for i, piece in enumerate(pieces):
        sides_eq = drawing_sides(piece)



        #piece.set_4_sides_eq(sides_eq[0],sides_eq[1],sides_eq[2],sides_eq[3])

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("All sides equations saved ! ")
    return ()





import os
os.environ['OMP_NUM_THREADS'] = '1'


def find_color(puzzle, window=True, k_dominant_colors=5):
    from sklearn.cluster import KMeans
    import numpy as np
    import cv2

    pieces = puzzle.get_pieces()

    def get_dominant_colors(pixels, k):
        if len(pixels) == 0:
            return np.zeros((k, 3), dtype=int), np.zeros(k)
        pixels = np.array(pixels)
        k = min(k, len(pixels))
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels)
        weights = counts / counts.sum()
        return centers, weights

    # Fonction pour échantillonner les pixels autour d'un rayon
    def sample_pixels_around_center(img, center, radius):
        pixels = []
        cx, cy = center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    # Si le pixel n'est pas noir
                    if not np.all(img[y, x] == [0, 0, 0]):
                        pixels.append(img[y, x])
        return pixels

    def sample_n_points_along_contour(contour, n_points):
        contour = np.array(contour, dtype=np.float32)

        # Filtrer les doublons
        filtered_contour = [contour[0]]
        for pt in contour[1:]:
            if np.linalg.norm(pt - filtered_contour[-1]) > 1e-5:
                filtered_contour.append(pt)
        contour = np.array(filtered_contour)

        if len(contour) < 2:
            raise ValueError("Le contour doit contenir au moins deux points distincts.")

        distances = [0]
        for i in range(1, len(contour)):
            d = np.linalg.norm(contour[i] - contour[i - 1])
            distances.append(distances[-1] + d)

        total_length = distances[-1]
        step = total_length / (n_points - 1)

        sampled_points = []
        for i in range(n_points):
            target_dist = i * step
            for j in range(1, len(distances)):
                segment_length = distances[j] - distances[j - 1]
                if segment_length == 0:
                    continue
                if distances[j] >= target_dist:
                    ratio = (target_dist - distances[j - 1]) / segment_length
                    point = (1 - ratio) * contour[j - 1] + ratio * contour[j]
                    sampled_points.append(tuple(map(int, point)))
                    break

        return sampled_points, step

    for idx, piece in enumerate(pieces):
        img = piece.get_color_image().copy()

        for side_idx, side in enumerate(piece.get_sides()):
            side_contour = side.get_side_contour()
            sampled_points, step = sample_n_points_along_contour(side_contour, n_points=18)
            radius = int(step / 2)
            #print("Radius :",radius,"\n\n")

            dominant_colors_per_side = []

            for pt in sampled_points:
                x, y = pt


                # Échantillonner les pixels autour du point (x, y)
                colors = sample_pixels_around_center(img, (x, y), radius)

                # Calcul des couleurs dominantes
                dom_colors, weights = get_dominant_colors(colors, k_dominant_colors)

                # Trouver la couleur dominante (celle avec le poids maximal)
                max_weight_idx = np.argmax(weights)  # Index du poids maximum
                dominant_color = dom_colors[max_weight_idx]  # Couleur associée au poids maximum

                # Stocker uniquement la couleur dominante pour ce côté
                dominant_colors_per_side.append(dominant_color[::-1])

                # Optionnel: affichage des couleurs dominantes dans l'image
                if window:
                    dominant_color = tuple(map(int, dominant_color))  # Convertir en tuple d'entiers
                    cv2.circle(img, (x, y), 10, dominant_color, -1)


            # Ajouter la couleur dominante calculée au côté
            side.set_side_color2(dominant_colors_per_side)

        # Affichage dans une fenêtre si demandé
        if window:
            #cv2.imshow(f"Piece {idx} - original", piece.get_color_image().copy())
            cv2.imshow(f"Piece {idx + 1}", img)
            cv2.waitKey(0)  # Attente de touche pour fermer la fenêtre
            cv2.destroyAllWindows()

    return puzzle

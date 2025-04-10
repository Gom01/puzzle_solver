import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from skimage.color import rgb2lab

from Processing_puzzle import Puzzle as p
from Processing_puzzle import Tools as t


def sides_fit(side1,color1, side2,color2):
    if color1 != 2 and color2 != 2:
        # Convertir en arrays
        color1 = np.array(color1)
        color2 = np.array(color2)

        # Calcul de la taille cible
        l1, l2 = len(color1), len(color2)
        l = min(l1, l2)

        # Redimensionner les deux listes à la même taille par interpolation
        def resize_colors(color_array, new_length):
            x_old = np.linspace(0, 1, len(color_array))
            x_new = np.linspace(0, 1, new_length)
            resized = np.zeros((new_length, 3))
            for i in range(3):  # R, G, B
                f = interp1d(x_old, color_array[:, i], kind='linear')
                resized[:, i] = f(x_new)
            return resized

        color1_resized = resize_colors(color1, l)
        color2_resized = resize_colors(color2, l)

        # --- 1. MSE + score de similarité ---
        mse = np.mean((color1_resized - color2_resized) ** 2)
        note_mse = 1 / (1 + mse)

        # --- 2. Corrélation RGB moyenne ---
        corrs = []
        for i in range(3):  # R, G, B
            corr, _ = pearsonr(color1_resized[:, i], color2_resized[:, i])
            corrs.append(corr)
        note_corr = np.mean(corrs)

        # --- 3. Distance perceptuelle (LAB ΔE) ---
        # Normaliser entre 0-1 si besoin
        if np.max(color1_resized) > 1.0:
            color1_resized /= 255
            color2_resized /= 255

        lab1 = rgb2lab(color1_resized.reshape(1, -1, 3))[0]
        lab2 = rgb2lab(color2_resized.reshape(1, -1, 3))[0]

        deltaE = np.linalg.norm(lab1 - lab2, axis=1)
        mean_deltaE = np.mean(deltaE)
        note_perceptuelle = 1 / (1 + mean_deltaE)

        print(f"note_mse : {note_mse:.4f}")
        print(f"note_correlation_rgb : {note_corr:.4f}")
        print(f"note_perceptuelle_lab : {note_perceptuelle:.4f}")

    return (side1 == 1 and side2 == -1) or (side1 == -1 and side2 == 1) or (side1 == 2 or side2 == 2)


def rotate_piece(piece, angle=90):  # Default angle is 90 degrees
    if piece.get_sides_info()[0] != 2:
        s1, s2, s3, s4 = piece.get_sides_info()
        piece.increment_number_rotation()
        c1,c2,c3,c4 = piece.get_sides_color()
        piece.set_sides_info(s4, s1, s2, s3)
        piece.set_sides_color(c4, c1, c2, c3)

    # Get the image and rotate it
    #img = piece.get_color_image()  # Assuming this gives you a PIL Image
    #rotated_img = t.Tools().rotate_image(img, angle)

    # Set the rotated image back to the piece
    #piece.set_color_image(rotated_img)


def print_grid(grid):
    for row in grid:
        print([str(piece) if piece else "None" for piece in row])


def classify_pieces(pieces):
    corners, borders, insides, wrongs = [], [], [], []
    for piece in pieces:
        sides = piece.get_sides_info()
        if not sides:
            piece.set_sides_info(2, 2, 2, 2)
            piece.set_sides_color(2,2,2,2)
            wrongs.append(piece)
            continue
        zeros = sides.count(0)
        if zeros == 2:
            corners.append(piece)
        elif zeros == 1:
            borders.append(piece)
        else:
            insides.append(piece)
    return corners, borders, insides, wrongs


def is_corner_position(row, col):
    return (row, col) in [(0, 0), (0, 5), (3, 0), (3, 5)]


def is_border_position(row, col):
    if row in [0, 3] and 1 <= col <= 4:
        return True
    if col in [0, 5] and 1 <= row <= 2:
        return True
    return False


def is_inside_position(row, col):
    return 1 <= row <= 2 and 1 <= col <= 4


def can_place(piece, grid, row, col, corners, borders, insides):
    # Check position constraints
    if piece in corners and not is_corner_position(row, col):
        return False
    if piece in borders and not is_border_position(row, col):
        return False
    if piece in insides and not is_inside_position(row, col):
        return False

    s1, s2, s3, s4 = piece.get_sides_info()  # [left, top, right, down]
    c1, c2, c3, c4 = piece.get_sides_color() # [left, top, right, down]

    # Check edge constraints
    if row == 0 and s2 != 0:
        return False
    if col == 0 and s1 != 0:
        return False
    if row == 3 and s4 != 0:
        return False
    if col == 5 and s3 != 0:
        return False

    # Check top neighbor
    if row > 0 and grid[row - 1][col] is not None:
        if not sides_fit(s2,c2, grid[row - 1][col].get_sides_info()[3],grid[row - 1][col].get_sides_color()[3]):  # match with neighbor's bottom
            return False

    # Check bottom neighbor
    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        if not sides_fit(s4, c4, grid[row + 1][col].get_sides_info()[1],grid[row + 1][col].get_sides_color()[1]):  # match with neighbor's top
            return False

    # Check left neighbor
    if col > 0 and grid[row][col - 1] is not None:
        if not sides_fit(s1, c1, grid[row][col - 1].get_sides_info()[2], grid[row][col - 1].get_sides_color()[2]):  # match with neighbor's right
            return False

    # Check right neighbor
    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        if not sides_fit(s3, c3,  grid[row][col + 1].get_sides_info()[0], grid[row][col + 1].get_sides_color()[0]):  # match with neighbor's left
            return False

    return True


def solve(grid, pieces, corners, borders, insides, row=0, col=0):
    print_grid(grid)
    if len(pieces) == 0:
        return True
    if col == 6:
        return solve(grid, pieces, corners, borders, insides, row + 1, 0)
    if row == 4:
        return False

    for piece in pieces[:]:  # copy to allow removal
        for _ in range(4):
            if can_place(piece, grid, row, col, corners, borders, insides):
                grid[row][col] = piece
                pieces.remove(piece)

                if solve(grid, pieces, corners, borders, insides, row, col + 1):
                    return True

                grid[row][col] = None
                pieces.append(piece)
            rotate_piece(piece)
    return False


def build_image(solution_indices, pieces, scale_factor=0.2):
    # Calculate the maximum width and height of all the pieces
    max_width, max_height = 0, 0
    for row in solution_indices:
        for piece in row:
            piece.rotate()
            img = piece.get_color_image()  # Get the image for the current piece
            img = np.array(img)  # Convert PIL image to OpenCV format (numpy array)
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)  # Resize piece
            max_width = max(max_width, img.shape[1])
            max_height = max(max_height, img.shape[0])

    # Create a blank canvas for the final image with the calculated dimensions
    final_image = np.ones((max_height * len(solution_indices), max_width * len(solution_indices[0]), 3), dtype=np.uint8) * 255

    # Create the final image by placing each piece on the canvas
    for row_idx, row in enumerate(solution_indices):
        for col_idx, piece in enumerate(row):
            img = piece.get_color_image()  # Get the image for the current piece
            img = np.array(img)  # Convert PIL image to OpenCV format (numpy array)
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)  # Resize piece

            # Calculate where to place this piece on the final image
            x_offset = col_idx * max_width + (max_width - img.shape[1]) // 2
            y_offset = row_idx * max_height + (max_height - img.shape[0]) // 2

            # Paste the piece onto the final image
            final_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    return final_image


def main():
    path = "../res/puzzle.pickle"
    puzzle = p.Puzzle()
    puzzle.load_puzzle(path)
    pieces = puzzle.get_pieces()

    corners, borders, insides, wrongs = classify_pieces(pieces)
    print(f"Corners: {len(corners)}")
    print(f"Borders: {len(borders)}")
    print(f"Insides: {len(insides)}")
    print(f"Wrongs: {len(wrongs)}")

    grid = [[None for _ in range(6)] for _ in range(4)]

    # Place a corner at top-left (0, 0)
    s1, s2, s3, s4 = corners[0].get_sides_info()
    while s1 != 0 or s2 != 0:
        rotate_piece(corners[0])
        s1, s2, s3, s4 = corners[0].get_sides_info()

    grid[0][0] = corners.pop(0)

    all_pieces = corners + borders + insides + wrongs

    print("Solving...")
    if solve(grid, all_pieces, corners, borders, insides, 0, 1):
        print("Solved!")
    else:
        print("Failed to solve.")

    print_grid(grid)

    final_image = build_image(grid, all_pieces)

    # Show the result
    cv2.imshow("Solved Puzzle", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

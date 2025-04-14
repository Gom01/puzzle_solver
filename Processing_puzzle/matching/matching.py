import numpy as np
import cv2
from tifffile import imshow

from Processing_puzzle import Puzzle as p
from Processing_puzzle import Tools as t
from Processing_puzzle.Sides import Side
import numpy as np

def compute_fit_score(piece, grid, row, col):
    score = 0
    sides = piece.get_sides()

    def calc_score(side1, side2):
        if not sides_fit(side1, side2):
            return -10

        size1 = side1.get_side_size()
        size2 = side2.get_side_size()

        if size1 == 2 or size2 == 2:
            base_score = -5
        else:
            diff = abs(size1 - size2)
            max_size = max(size1, size2)
            size_score = max(0, 1 - (diff / max_size))
            base_score = size_score*5
        # Color matching
        colors1 = np.array(side1.get_side_color())  # shape: (3, 3)
        colors2 = np.array(side2.get_side_color())  # shape: (3, 3)
        # Compare each color from one side to all on the other and take the best match
        total_color_score = 0
        for c1 in colors1:
            dists = np.linalg.norm(colors2 - c1, axis=1)  # Euclidean distances
            best_match_score = 1 - min(dists) / 255  # normalize, lower distance = better
            total_color_score += best_match_score

        color_score = total_color_score / len(colors1)
        color_bonus = color_score * 2  # weight of color in final score

        return base_score + color_bonus

    # Top neighbor
    if row > 0 and grid[row - 1][col] is not None:
        score += calc_score(sides[3], grid[row - 1][col].get_sides()[1])
    else: score += 6

    # Bottom neighbor
    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        score += calc_score(sides[1], grid[row + 1][col].get_sides()[3])
    else:
        score += 6


    # Left neighbor
    if col > 0 and grid[row][col - 1] is not None:
        score += calc_score(sides[0], grid[row][col - 1].get_sides()[2])
    else:
        score += 6

    # Right neighbor
    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        score += calc_score(sides[2], grid[row][col + 1].get_sides()[0])
    else:
        score += 6

    return score





def sides_fit(side1, side2):
    inf1 = side1.get_side_info()
    inf2 = side2.get_side_info()
    if inf1 == 1 and inf2 == -1:
        return True
    elif inf1 == -1 and inf2 == 1:
        return True
    elif inf1 == 2 or inf2 == 2:
        return True
    else:
        return False

#gacuche , bottom right up
def rotate_piece(piece):
    piece.increment_number_rotation()
    s0,s1,s2,s3 = piece.get_sides()
    piece.set_sides(s1, s2, s3, s0)  # Rotate 90° clockwise


def print_grid(grid):
    for row in grid:
        print([str(piece) if piece else "None" for piece in row])


def classify_pieces(pieces):
    corners, borders, insides, wrongs = [], [], [], []

    for piece in pieces:
        sides = piece.get_sides()
        if piece.is_bad:
            for s in piece.get_sides():
                s.set_side_info(2)
            wrongs.append(piece)
            continue
        count = sum(1 for side in sides if side.get_side_info() == 0)
        if count == 2:
            corners.append(piece)
        elif count == 1:
            borders.append(piece)
        else:
            insides.append(piece)
    return corners, borders, insides, wrongs


def is_corner_position(row, col):
    return (row, col) in [(0, 5), (3, 0), (3, 5)]


def is_border_position(row, col):
    return row == 0 or row == 3 or col == 0 or col == 5


def is_inside_position(row, col):
    return 1 <= row <= 2 and 1 <= col <= 4


def can_place(piece, grid, row, col, corners, borders, insides):
    if piece in corners and not is_corner_position(row, col):
        #print("Not corner")
        return False
    elif piece in borders and not is_border_position(row, col):
        #print("Not border")
        return False
    elif piece in insides and not is_inside_position(row, col):
        #print("Not inside")
        return False

    s1, s2, s3, s4 = piece.get_sides_info()  # [left, bottom, right, top]

    if row == 0 and (s4 == 1 or s4 == -1):  # top
        #print("Zero not correct top")
        return False
    if col == 0 and (s1 == 1 or s1 == -1):  # left
        #print("Zero not correct left")
        return False
    if row == 3 and (s2 == 1 or s2 == -1):  # bottom
        #print("Zero not correct bottom")
        return False
    if col == 5 and (s3 == 1 or s3 == -1):  # right
        #print("Zero not correct right")
        return False

    sides = piece.get_sides()

    if row > 0 and grid[row - 1][col] is not None:
        #print("Check top neighbor")
        if not sides_fit(sides[3], grid[row - 1][col].get_sides()[1]):
            return False

    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        #print("Check bottom neighbor")
        if not sides_fit(sides[1], grid[row + 1][col].get_sides()[3]):
            return False

    if col > 0 and grid[row][col - 1] is not None:
        #print("Check left neighbor")
        if not sides_fit(sides[0], grid[row][col - 1].get_sides()[2]):
            return False

    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        #print("Check right neighbor")
        if not sides_fit(sides[2], grid[row][col + 1].get_sides()[0]):
            return False

    return True


def build_frame(grid, corners, borders, wrongs):
    frame_path = [
        *[(0, col) for col in range(1, 6)],
        *[(row, 5) for row in range(1, 4)],
        *[(3, col) for col in range(4, -1, -1)],
        *[(row, 0) for row in range(2, 0, -1)]
    ]

    first_corner = corners.pop(0)


    for _ in range(4):
        sides = first_corner.get_sides()
        if sides[0].get_side_info() == 0 and sides[3].get_side_info() == 0:
            grid[0][0] = first_corner
            break
        rotate_piece(first_corner)

    # print("(0,0)")
    # print(first_corner.get_sides_info())
    # print()

    def backtrack(index, remaining_corners, remaining_borders, remaining_wrongs):
        if index >= len(frame_path):
            return True
        row, col = frame_path[index]
        # print(f"\nTrying position: ({row}, {col})")
        best_rotation = 0
        all_candidates = remaining_borders + remaining_corners + remaining_wrongs
        scored_candidates = []
        for piece in all_candidates:
            original = piece
            best_score = float('-inf')
            for r in range(4):
                score = compute_fit_score(piece, grid, row, col)
                if can_place(piece, grid, row, col, remaining_corners, remaining_borders, remaining_wrongs):
                    best_score = max(best_score, score)
                    best_rotation = r
                rotate_piece(piece)
            piece.reset_piece(original)
            scored_candidates.append((piece, best_score))

        candidates = [p for p, _ in sorted(scored_candidates, key=lambda x: x[1], reverse=True)]

        for piece in candidates:
            original_piece = piece
            for _ in range(4):
                if can_place(piece, grid, row, col, remaining_corners, remaining_borders, []):
                    grid[row][col] = piece
                    new_corners = remaining_corners.copy()
                    new_borders = remaining_borders.copy()
                    new_wrongs = remaining_wrongs.copy()

                    if piece in remaining_corners:
                        new_corners.remove(piece)
                    elif piece in remaining_borders:
                        new_borders.remove(piece)
                    else:
                        new_wrongs.remove(piece)

                    if backtrack(index + 1, new_corners, new_borders, new_wrongs):
                        return True

                    grid[row][col] = None
                rotate_piece(piece)

            piece.reset_piece(original_piece)
        return False

    success = backtrack(0, corners.copy(), borders.copy(), wrongs.copy())
    if not success:
        print("❌ Could not build the frame.")
    return success


def build_inside(grid, insides, wrongs):
    # Define the positions for the inside pieces: rows 1 and 2, cols 1 to 4
    frame_path = [(row, col) for row in range(1, 3) for col in range(1, 5)]

    def backtrack(index, remaining_insides, remaining_wrongs):
        if index >= len(frame_path):
            return True
        row, col = frame_path[index]
        # print(f"\n🔄 Trying position: ({row}, {col})")

        all_candidates = remaining_insides + remaining_wrongs
        scored_candidates = []

        for piece in all_candidates:
            original = piece
            best_score = float('-inf')
            for _ in range(4):
                score = compute_fit_score(piece, grid, row, col)
                if can_place(piece, grid, row, col, [], [], all_candidates):
                    best_score = max(best_score, score)
                rotate_piece(piece)
            piece.reset_piece(original)
            scored_candidates.append((piece, best_score))


        candidates = [p for p, _ in sorted(scored_candidates, key=lambda x: x[1], reverse=True)]


        for piece in candidates:
            original_piece = piece
            for _ in range(4):
                if can_place(piece, grid, row, col, [], [], remaining_insides + remaining_wrongs):
                    grid[row][col] = piece
                    new_insides = remaining_insides.copy()
                    new_wrongs = remaining_wrongs.copy()

                    if piece in new_insides:
                        new_insides.remove(piece)
                    else:
                        new_wrongs.remove(piece)

                    if backtrack(index + 1, new_insides, new_wrongs):
                        return True

                    grid[row][col] = None
                rotate_piece(piece)

            piece.reset_piece(original_piece)  # Reset rotation and sides
        return False

    success = backtrack(0, insides.copy(), wrongs.copy())
    if not success:
        print("❌ Could not place all inside pieces.")
    return success



def build_image(solution_indices, pieces, scale_factor=0.2):
    max_width, max_height = 0, 0

    # First pass: determine maximum width and height after rotation + resizing
    for row in solution_indices:
        for piece in row:
            if piece is not None:
                rotated_img = piece.rotate_image_by_rotation()  # Make sure this returns the rotated image without resetting
                img = cv2.resize(rotated_img, (0, 0), fx=scale_factor, fy=scale_factor)
                max_width = max(max_width, img.shape[1])
                max_height = max(max_height, img.shape[0])

    # Create final canvas
    final_image = np.ones((max_height * len(solution_indices), max_width * len(solution_indices[0]), 3),
                          dtype=np.uint8) * 255

    # Second pass: paste each image in its slot
    for row_idx, row in enumerate(solution_indices):
        for col_idx, piece in enumerate(row):
            if piece is not None:
                # Get rotated image again
                rotated_img = piece.rotate_image_by_rotation()
                img = cv2.resize(rotated_img, (0, 0), fx=scale_factor, fy=scale_factor)
            else:
                img = np.zeros((max_height, max_width, 3), dtype=np.uint8)

            # Compute position with centering
            x_offset = col_idx * max_width + (max_width - img.shape[1]) // 2
            y_offset = row_idx * max_height + (max_height - img.shape[0]) // 2

            # Safe bounds check
            if (y_offset + img.shape[0] <= final_image.shape[0]) and (x_offset + img.shape[1] <= final_image.shape[1]):
                final_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
            else:
                print(f"Skipping piece at ({row_idx}, {col_idx}) due to size mismatch.")
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


    print("\n🎯 Starting to build frame...\n")
    success = build_frame(grid, corners, borders,wrongs)
    if not success:
        print("Failed to build frame.")
        return
    print("✅ Frame built.")
    print_grid(grid)



    print("\n🎯 Starting to place inside pieces...\n")
    if not build_inside(grid, insides, wrongs):
        print("Failed ! ")
    else:
        print("✅ Filled missing pieces with wrong pieces.")

    print_grid(grid)
    final_image = build_image(grid, corners + borders + insides + wrongs)
    cv2.imshow("Solved Puzzle", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

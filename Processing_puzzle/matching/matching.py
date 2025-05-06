from unittest.util import sorted_list_difference
import numpy as np
import cv2
from Processing_puzzle import Puzzle as p
from test_sides import side_similarities
from side_shape import color_similarities

def compute_fit_score(piece, grid, row, col):
    score = 0
    sides = piece.get_sides()

    def calc_score(side1, side2):

        s1, s2 = side1.get_side_info(), side2.get_side_info()

        #colors1, weight1 = side1.get_side_color()
        #colors2, weight2 = side2.get_side_color()
        #color_score = color_similarities(colors1, weight1, colors2, weight2)

        #if s1 == 2 or s2 == 2:
        #    return color_score - 2

        confidence = side_similarities(side1, side2, True)

        #size1 = side1.get_side_size()
        #size2 = side2.get_side_size()
        #diff = abs(size1 - size2)
        #max_size = max(size1, size2)
        #size_score = max(0, 1 - (diff / max_size))

        return confidence


    # Check adjacent pieces and compute the score
    if row > 0 and grid[row - 1][col] is not None:
        score += calc_score(sides[3], grid[row - 1][col].get_sides()[1])

    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        score += calc_score(sides[1], grid[row + 1][col].get_sides()[3])


    if col > 0 and grid[row][col - 1] is not None:
        score += calc_score(sides[0], grid[row][col - 1].get_sides()[2])

    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        score += calc_score(sides[2], grid[row][col + 1].get_sides()[0])


    print(f"{piece.index} : final score = {score}")

    return score



def sides_fit(side1, side2):
    inf1 = side1.get_side_info()
    inf2 = side2.get_side_info()
    return (inf1 == 1 and inf2 == -1) or (inf1 == -1 and inf2 == 1) or inf1 == 2 or inf2 == 2

def rotate_piece(piece):
    piece.increment_number_rotation()
    s0, s1, s2, s3 = piece.get_sides()
    piece.set_sides(s1, s2, s3, s0)

def print_grid(grid):
    for row in grid:
        print([str(piece) if piece else "None" for piece in row])

def classify_pieces(pieces):
    corners, borders, insides, wrongs = [], [], [], []
    for piece in pieces:
        sides = piece.get_sides()
        if piece.is_bad:
            for s in sides:
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


def can_place(piece, grid, row, col, corners, borders, insides, wrongs):
    corner_positions = {(0, 0), (0, 5), (3, 0), (3, 5)}
    border_positions = {
        *[(0, c) for c in range(1, 5)],
        *[(3, c) for c in range(1, 5)],
        *[(r, 0) for r in range(1, 3)],
        *[(r, 5) for r in range(1, 3)]
    }
    inside_positions = {(r, c) for r in range(1, 3) for c in range(1, 5)}

    if (row, col) in corner_positions:
        if piece not in corners and piece not in wrongs:
            return False
    elif (row, col) in border_positions:
        if piece not in borders and piece not in wrongs:
            return False
    elif (row, col) in inside_positions:
        if piece not in insides and piece not in wrongs:
            return False

    s1, s2, s3, s4 = piece.get_sides_info()  # [left, bottom, right, top]
    if row == 0 and s4 not in (0, 2): return False  # top edge
    if row == 3 and s2 not in (0, 2): return False  # bottom edge
    if col == 0 and s1 not in (0, 2): return False  # left edge
    if col == 5 and s3 not in (0, 2): return False  # right edge

    sides = piece.get_sides()

    # Check top
    if row > 0 and grid[row - 1][col] is not None:
        if not sides_fit(sides[3], grid[row - 1][col].get_sides()[1]):
            return False
    # Check bottom
    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        if not sides_fit(sides[1], grid[row + 1][col].get_sides()[3]):
            return False
    # Check left
    if col > 0 and grid[row][col - 1] is not None:
        if not sides_fit(sides[0], grid[row][col - 1].get_sides()[2]):
            return False
    # Check right
    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        if not sides_fit(sides[2], grid[row][col + 1].get_sides()[0]):
            return False

    return True


def solve_puzzle(grid, corners, borders, insides, wrongs):
    def build_path():
        frame = [(0, 0)]
        frame += [(0, col) for col in range(1, 6)]
        frame += [(row, 5) for row in range(1, 4)]
        frame += [(3, col) for col in range(4, -1, -1)]
        frame += [(row, 0) for row in range(2, 0, -1)]
        inside = [(r, c) for r in range(1, 3) for c in range(1, 5)]
        return frame + inside

    full_path = build_path()
    total_slots = len(full_path)

    first_row, first_col = full_path[0]
    first_piece = corners[0]

    # Orient the first corner piece
    for _ in range(4):
        s1, s2, s3, s4 = first_piece.get_sides_info()
        if s1 == 0 and s4 == 0:
            break
        rotate_piece(first_piece)

    grid[first_row][first_col] = first_piece
    new_corners = corners.copy()
    new_corners.remove(first_piece)

    # Define the corner, border, and inside positions
    corner_positions = {(0,0),(0, 5), (3, 0), (3, 5)}
    border_positions = {
        *[(0, c) for c in range(1, 5)],
        *[(3, c) for c in range(1, 5)],
        *[(r, 0) for r in range(1, 3)],
        *[(r, 5) for r in range(1, 3)]
    }
    inside_positions = {(r, c) for r in range(1, 3) for c in range(1, 5)}

    def backtrack(index, corners_left, borders_left, insides_left, wrongs_left):


        print(f"\n🧩 Progress: {index}/{total_slots} | Corners: {len(corners_left)} | Borders: {len(borders_left)} | Insides: {len(insides_left)} | Wrongs: {len(wrongs_left)}")
        print_grid(grid)

        if index >= total_slots:
            return True

        row, col = full_path[index]
        is_frame = index < len(full_path) - 8  # 24 - 16 = 8 inside slots

        if is_frame:
            if (row, col) in corner_positions:
                candidates = corners_left + wrongs_left
            elif (row, col) in border_positions:
                candidates = borders_left + wrongs_left
            else:
                candidates = wrongs_left
        else:
            if len(corners_left) != 0 or len(borders_left) != 0:
                return False
            candidates = insides_left + wrongs_left

        scored_candidates = []

        for piece in candidates:
            original = piece.clone() if hasattr(piece, "clone") else piece
            best_score, best_rotation = float('-inf'), 0

            for r in range(4):
                if can_place(piece, grid, row, col, corners_left, borders_left, insides_left, wrongs_left):
                    score = compute_fit_score(piece, grid, row, col)
                    if score > best_score:
                        best_score, best_rotation = score, r
                rotate_piece(piece)

            piece.reset_piece(original)
            scored_candidates.append((piece, best_score, best_rotation))

        # Sort pieces by fit score (higher is better)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        print(scored_candidates)


        for piece, score, rotation in scored_candidates:
            original = piece.clone() if hasattr(piece, "clone") else piece
            for _ in range(rotation):
                rotate_piece(piece)

            if can_place(piece, grid, row, col, corners_left, borders_left, insides_left, wrongs_left):
                print(f"✅ Trying {piece} at ({row}, {col}) with rotation {rotation} and score {score:.2f}")
                grid[row][col] = piece

                # Track and remove the placed piece from the lists
                lists = {'c': corners_left.copy(), 'b': borders_left.copy(), 'i': insides_left.copy(), 'w': wrongs_left.copy()}
                piece_type = next((k for k, lst in lists.items() if piece in lst), None)
                if piece_type:
                    lists[piece_type].remove(piece)

                if backtrack(index + 1, lists['c'], lists['b'], lists['i'], lists['w']):
                    return True

                # If placement fails, backtrack
                print(f"↩️ Backtracking from ({row}, {col})")
                grid[row][col] = None
                piece.reset_piece(original)

        return False  # If no piece fits

    success = backtrack(1, new_corners, borders.copy(), insides.copy(), wrongs.copy())
    print("✅ Puzzle solved!" if success else "❌ No solution found.")
    return success


def build_image(solution_indices, pieces, scale_factor=0.2):
    max_width, max_height = 0, 0

    # First pass: determine max width/height for grid layout
    for row in solution_indices:
        for piece in row:
            if piece is not None:
                rotated_img = piece.rotate_image_by_rotation()
                img = cv2.resize(rotated_img, (0, 0), fx=scale_factor, fy=scale_factor)
                max_width = max(max_width, img.shape[1])
                max_height = max(max_height, img.shape[0])

    final_image = np.ones((max_height * len(solution_indices), max_width * len(solution_indices[0]), 3), dtype=np.uint8) * 255

    # Second pass: place each image and add text
    for row_idx, row in enumerate(solution_indices):
        for col_idx, piece in enumerate(row):
            if piece is not None:
                rotated_img = piece.rotate_image_by_rotation()
                img = cv2.resize(rotated_img, (0, 0), fx=scale_factor, fy=scale_factor)
                index = piece.get_index()
            else:
                img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                index = None
                print(f"Piece missing at ({row_idx}, {col_idx}) - No image to place.")

            x_offset = col_idx * max_width + (max_width - img.shape[1]) // 2
            y_offset = row_idx * max_height + (max_height - img.shape[0]) // 2

            if (y_offset + img.shape[0] <= final_image.shape[0]) and (x_offset + img.shape[1] <= final_image.shape[1]):
                final_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

                # Draw index if piece exists
                if index is not None:
                    text = f"{index}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = x_offset + (img.shape[1] - text_size[0]) // 2
                    text_y = y_offset + text_size[1] + 10
                    cv2.putText(final_image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
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

    print("\n🎯 Starting full puzzle solving...\n")
    if solve_puzzle(grid, corners, borders, insides, wrongs):
        print("✅ Puzzle solved.")
    else:
        print("❌ Puzzle could not be solved.")

    print_grid(grid)

    final_image = build_image(grid, pieces)
    cv2.imshow("Solved Puzzle", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

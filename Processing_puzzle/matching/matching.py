import numpy as np
import cv2
from Processing_puzzle import Puzzle as p
from Processing_puzzle import Tools as t
from Processing_puzzle.Sides import Side


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


def rotate_piece(piece):
    piece.increment_number_rotation()
    sides = piece.get_sides()
    # New order: [left, bottom, right, top]
    piece.set_sides(sides[3], sides[0], sides[1], sides[2])  # Rotate 90° clockwise


def print_grid(grid):
    for row in grid:
        print([str(piece) if piece else "None" for piece in row])


def classify_pieces(pieces):
    corners, borders, insides, wrongs = [], [], [], []
    for piece in pieces:
        sides = piece.get_sides()
        if piece.is_bad:
            piece.set_sides(Side(2), Side(2), Side(2), Side(2))
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
        print("Not corner")
        return False
    elif piece in borders and not is_border_position(row, col):
        print("Not border")
        return False
    elif piece in insides and not is_inside_position(row, col):
        print("Not inside")
        return False

    s1, s2, s3, s4 = piece.get_sides_info()  # [left, bottom, right, top]

    if row == 0 and s4 != 0:  # top
        print("Zero not correct top")
        return False
    if col == 0 and s1 != 0:  # left
        print("Zero not correct left")
        return False
    if row == 3 and s2 != 0:  # bottom
        print("Zero not correct bottom")
        return False
    if col == 5 and s3 != 0:  # right
        print("Zero not correct right")
        return False

    sides = piece.get_sides()

    if row > 0 and grid[row - 1][col] is not None:
        print("Check top neighbor")
        if not sides_fit(sides[3], grid[row - 1][col].get_sides()[1]):
            return False

    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        print("Check bottom neighbor")
        if not sides_fit(sides[1], grid[row + 1][col].get_sides()[3]):
            return False

    if col > 0 and grid[row][col - 1] is not None:
        print("Check left neighbor")
        if not sides_fit(sides[0], grid[row][col - 1].get_sides()[2]):
            return False

    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        print("Check right neighbor")
        if not sides_fit(sides[2], grid[row][col + 1].get_sides()[0]):
            return False

    return True


def build_frame(grid, corners, borders):
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

    print("(0,0)")
    print(first_corner.get_sides_info())
    print()

    def backtrack(index, remaining_corners, remaining_borders):
        candidates = remaining_borders + remaining_corners
        if index >= len(frame_path):
            return True
        if len(candidates) == 0:
            return True

        row, col = frame_path[index]
        print(f"\nTrying position: ({row}, {col})")

        for piece in candidates:
            original_piece = piece
            for _ in range(4):
                if can_place(piece, grid, row, col, remaining_corners, remaining_borders, []):
                    grid[row][col] = piece
                    new_corners = remaining_corners.copy()
                    new_borders = remaining_borders.copy()

                    if piece in remaining_corners:
                        new_corners.remove(piece)
                    else:
                        new_borders.remove(piece)

                    if backtrack(index + 1, new_corners, new_borders):
                        return True

                    grid[row][col] = None
                rotate_piece(piece)

            piece.reset_piece(original_piece)
        return False

    success = backtrack(0, corners.copy(), borders.copy())
    if not success:
        print("❌ Could not build the frame.")
    return success


def solve(grid, pieces, corners, borders, insides, row=0, col=0):
    print_grid(grid)
    if len(pieces) == 0:
        return True
    if col == 6:
        return solve(grid, pieces, corners, borders, insides, row + 1, 0)
    if row == 4:
        return False

    for piece in pieces[:]:
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
    max_width, max_height = 0, 0
    for row in solution_indices:
        for piece in row:
            if piece is not None:
                piece.rotate()
                img = piece.get_color_image()
                img = cv2.resize(np.array(img), (0, 0), fx=scale_factor, fy=scale_factor)
                max_width = max(max_width, img.shape[1])
                max_height = max(max_height, img.shape[0])

    final_image = np.ones((max_height * len(solution_indices), max_width * len(solution_indices[0]), 3),
                          dtype=np.uint8) * 255

    for row_idx, row in enumerate(solution_indices):
        for col_idx, piece in enumerate(row):
            if piece is not None:
                img = cv2.resize(np.array(piece.get_color_image()), (0, 0), fx=scale_factor, fy=scale_factor)
            else:
                img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            x_offset = col_idx * max_width + (max_width - img.shape[1]) // 2
            y_offset = row_idx * max_height + (max_height - img.shape[0]) // 2
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

    build_frame(grid, corners, borders)
    print_grid(grid)

    final_image = build_image(grid, corners + borders)
    cv2.imshow("Solved Puzzle", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

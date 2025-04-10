import numpy as np
import cv2
from Processing_puzzle import Puzzle as p
from Processing_puzzle import Tools as t
from Processing_puzzle.Sides import Side


def sides_fit(side1, side2):
    inf1 = side1.get_side_info()
    inf2 = side2.get_side_info()
    return (inf1 == 1 and inf2 == -1) or (inf1 == -1 and inf2 == 1) or (inf1 == 2 or inf2 == 2)


def rotate_piece(piece, angle=90):
    sides = piece.get_sides()
    s1, s2, s3, s4 = sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info()
    if s1 != 2 or s2 != 2 or s3 != 2 or s4 != 2:
        piece.increment_number_rotation()
        piece.set_sides(sides[3], sides[0], sides[1], sides[2])


def print_grid(grid):
    for row in grid:
        print([str(piece) if piece else "None" for piece in row])


def classify_pieces(pieces):
    corners, borders, insides, wrongs = [], [], [], []
    for piece in pieces:
        sides = piece.get_sides()
        if not sides:
            piece.set_sides(Side(2), Side(2), Side(2), Side(2))
            sides_ = piece.get_sides()
            sides_[0].set_side_info(2)
            sides_[1].set_side_info(2)
            sides_[2].set_side_info(2)
            sides_[3].set_side_info(2)
            wrongs.append(piece)
            continue
        count = 0
        for side in sides:
            if side.get_side_info() == 0:
                count += 1
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
    if row in [0, 3] and 1 <= col <= 4:
        return True
    if col in [0, 5] and 1 <= row <= 2:
        return True
    return False


def is_inside_position(row, col):
    return 1 <= row <= 2 and 1 <= col <= 4


def can_place(piece, grid, row, col, corners, borders, insides):
    if piece in corners and not is_corner_position(row, col):
        return False
    if piece in borders and not is_border_position(row, col):
        return False
    if piece in insides and not is_inside_position(row, col):
        return False

    sides = piece.get_sides()
    s1, s2, s3, s4 = sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info() # [left, top, right, down]

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
        if not sides_fit(sides[1], grid[row - 1][col].get_sides()[3]):  # match with neighbor's bottom
            return False

    # Check bottom neighbor
    if row < len(grid) - 1 and grid[row + 1][col] is not None:
        if not sides_fit(sides[3], grid[row + 1][col].get_sides()[1]):  # match with neighbor's top
            return False

    # Check left neighbor
    if col > 0 and grid[row][col - 1] is not None:
        if not sides_fit(sides[0], grid[row][col - 1].get_sides()[2]):  # match with neighbor's right
            return False

    # Check right neighbor
    if col < len(grid[0]) - 1 and grid[row][col + 1] is not None:
        if not sides_fit(sides[2], grid[row][col + 1].get_sides()[0]):  # match with neighbor's left
            return False

    return True


def solve(grid, pieces, corners, borders, insides, row=0, col=0):
    print_grid(grid)
    if len(pieces) == 0:
        return True
    if col == 6:
        return solve(grid, pieces, corners, borders, insides, row + 1, 0)
    if row == 4:  # Fixed row boundary issue
        return False

    for piece in pieces[:]:  # Copy to allow removal
        for _ in range(4):  # Try 4 rotations
            if can_place(piece, grid, row, col, corners, borders, insides):
                grid[row][col] = piece
                pieces.remove(piece)

                # Move to the next position
                if solve(grid, pieces, corners, borders, insides, row, col + 1):
                    return True

                # Backtrack
                grid[row][col] = None
                pieces.append(piece)
            rotate_piece(piece)
    return False


def build_image(solution_indices, pieces, scale_factor=0.2):
    max_width, max_height = 0, 0
    for row in solution_indices:
        for piece in row:
            piece.rotate()  # Rotating piece before building image
            img = piece.get_color_image()
            img = np.array(img)  # Convert to numpy array for manipulation
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            max_width = max(max_width, img.shape[1])
            max_height = max(max_height, img.shape[0])

    # Creating final image canvas
    final_image = np.ones((max_height * len(solution_indices), max_width * len(solution_indices[0]), 3), dtype=np.uint8) * 255

    # Place each piece on the final image canvas
    for row_idx, row in enumerate(solution_indices):
        for col_idx, piece in enumerate(row):
            img = piece.get_color_image()
            img = np.array(img)
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

            x_offset = col_idx * max_width + (max_width - img.shape[1]) // 2
            y_offset = row_idx * max_height + (max_height - img.shape[0]) // 2

            final_image[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    return final_image


def main():
    path = "../res/puzzle.pickle"
    puzzle = p.Puzzle()
    puzzle.load_puzzle(path)
    pieces = puzzle.get_pieces()

    for idx, piece in enumerate(pieces):
        img = piece.get_color_image()
        piece.increment_number_rotation()
        piece.rotate()

    corners, borders, insides, wrongs = classify_pieces(pieces)
    print(f"Corners: {len(corners)}")
    print(f"Borders: {len(borders)}")
    print(f"Insides: {len(insides)}")
    print(f"Wrongs: {len(wrongs)}")

    grid = [[None for _ in range(6)] for _ in range(4)]

    # Place a corner at top-left (0, 0)
    sides = corners[0].get_sides()
    s1, s2, s3, s4 = sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info()
    while s1 != 0 or s2 != 0:
        rotate_piece(corners[0])
        sides = corners[0].get_sides()
        s1, s2, s3, s4 = sides[0].get_side_info(), sides[1].get_side_info(), sides[2].get_side_info(), sides[3].get_side_info()

    grid[0][0] = corners.pop(0)

    all_pieces = corners + borders + insides + wrongs

    print("Solving...")
    if solve(grid, all_pieces, corners, borders, insides, 0, 1):
        print("Solved!")
        print_grid(grid)
        final_image = build_image(grid, all_pieces)

        cv2.imshow("Solved Puzzle", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Failed to solve.")


if __name__ == "__main__":
    main()

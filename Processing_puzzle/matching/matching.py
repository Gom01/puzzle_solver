from Processing_puzzle import Puzzle as p
from Processing_puzzle.matching.side_curve import compare_curvature
#from Processing_puzzle.matching.side_size import size_similarities
from side_shape import side_similarities
from side_color import color_similarities2
from time import sleep
import matplotlib.pyplot as plt


def plot_candidate_scores(scored_candidates, row, col, index):
    names = [f"P{piece.get_index()}" for piece, score, _ in scored_candidates]
    scores = [score if score != float('-inf') else 0 for _, score, _ in scored_candidates]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, scores, color='skyblue')

    max_score = max(scores)
    plt.axhline(y=max_score, color='r', linestyle='--', label='Best Score')

    plt.xlabel('Piece ID')
    plt.ylabel('Fit Score')
    plt.title(f'Fit Scores for Candidates at Position ({row}, {col}) - Step {index}')
    plt.xticks(rotation=45)

    # Allow autoscaling of Y-axis (remove cap at 1.05)
    plt.ylim(0, max_score * 1.1 if max_score > 0 else 1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_grid(grid, scale=0.5):
    def crop_piece(img):
        if img is None:
            print("⚠️ Warning: Attempted to crop a None image.")
            return np.ones((50, 50, 3), dtype=np.uint8) * 255  # fallback blank image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    rows_images, max_width = [], 0
    for row in grid:
        cropped_imgs, heights = [], []
        for piece in row:
            if piece is not None:
                img = piece.rotate_image_by_rotation()
                if img is None:
                    print(f"❌ Piece {piece.get_index()} has no color image.")
                    continue
                img = crop_piece(img)
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

                # ✏️ Add piece index label
                piece_id = f"{piece.get_index()}"
                cv2.putText(img, piece_id, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, cv2.LINE_AA)

                cropped_imgs.append(img)
                heights.append(img.shape[0])
            else:
                cropped_imgs.append(None)
                heights.append(0)

        row_height = max(heights)
        row_pieces = []
        for img in cropped_imgs:
            if img is not None:
                h, w = img.shape[:2]
                pad_top = (row_height - h) // 2
                pad_bottom = row_height - h - pad_top
                img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
                row_pieces.append(img_padded)
            else:
                row_pieces.append(np.ones((row_height, 1, 3), dtype=np.uint8) * 255)

        row_img = np.hstack(row_pieces)
        rows_images.append(row_img)
        max_width = max(max_width, row_img.shape[1])

    full_height = sum(img.shape[0] for img in rows_images)
    final_img = np.ones((full_height, max_width, 3), dtype=np.uint8) * 255
    y_offset = 0
    for row_img in rows_images:
        final_img[y_offset:y_offset + row_img.shape[0], :row_img.shape[1]] = row_img
        y_offset += row_img.shape[0]

    cv2.imshow("Live Puzzle Assembly", final_img)
    cv2.waitKey(1)

def calc_score(side1, side2, window=False):
    s1, s2 = side1.get_side_info(), side2.get_side_info()

    def is_ambiguous_side(side):
        return side.get_side_info() == 2

    #Measuring score
    colors_array1 = side1.get_side_color2()
    colors_array2 = side2.get_side_color2()
    color_score = color_similarities2(colors_array1, colors_array2, window=False)
    confidence = side_similarities(side1, side2, False)

    if window:
        im1 = side1.get_piece_image().copy()
        im2 = side2.get_piece_image().copy()
        s1_c, s2_c = np.array(side1.get_side_contour()), np.array(side2.get_side_contour())

        # Compute contour centroids
        center1 = tuple(np.mean(s1_c, axis=0).astype(int))
        center2 = tuple(np.mean(s2_c, axis=0).astype(int))

        # Draw circles at the center of each contour
        cv2.circle(im1, center1, radius=10, color=(0, 0, 255), thickness=-1)  # Red dot
        cv2.circle(im2, center2, radius=10, color=(0, 0, 255), thickness=-1)

        # Resize to same height if needed
        if im1.shape[0] != im2.shape[0]:
            height = min(im1.shape[0], im2.shape[0])
            im1 = cv2.resize(im1, (int(im1.shape[1] * height / im1.shape[0]), height))
            im2 = cv2.resize(im2, (int(im2.shape[1] * height / im2.shape[0]), height))

        # Concatenate and show
        combined = np.hstack((im2, im1))
        cv2.imshow("Compared Sides with Contour Markers", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    ##Red and green puzzle (darker color)
    #puzzleB
    #base_score =   color_score*0.7 + confidence*0.6

    #puzzleA
    #Very different colors
    #base_score =  color_score * 0.7 + confidence * 0.7

    #puzzleC
    base_score = color_score * 0.7 + confidence * 0.6


    if is_ambiguous_side(side1) or is_ambiguous_side(side2):
        return base_score * 0.2  # minimal influence

    return base_score




def compute_fit_score(piece, grid, row, col):
    score = 0
    sides = piece.get_sides()
    neighbor_coords = [(row-1, col, 3, 1), (row+1, col, 1, 3), (row, col-1, 0, 2), (row, col+1, 2, 0)]

    for r, c, i, j in neighbor_coords:
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] is not None:
            neighbor = grid[r][c]
            # Skip influence if neighbor's side is ambiguous
            if neighbor.get_sides()[j].get_side_info() == 2:
                continue
            score += calc_score(sides[i], neighbor.get_sides()[j])

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
        if piece.get_strait_image() is not None:
            piece.set_color_image(piece.get_strait_image())
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
    height = len(grid)
    width = len(grid[0])

    corner_positions = {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1)
    }
    border_positions = {
        *[(0, c) for c in range(1, width - 1)],
        *[(height - 1, c) for c in range(1, width - 1)],
        *[(r, 0) for r in range(1, height - 1)],
        *[(r, width - 1) for r in range(1, height - 1)],
    }

    inside_positions = {
        (r, c) for r in range(1, height - 1) for c in range(1, width - 1)
    }

    if (row, col) in corner_positions:
        if piece not in corners and piece not in wrongs:
            return False
    elif (row, col) in border_positions:
        if piece not in borders and piece not in wrongs:
            return False
    elif (row, col) in inside_positions:
        if piece not in insides and piece not in wrongs:
            return False

    max_row = len(grid) - 1
    max_col = len(grid[0]) - 1

    s1, s2, s3, s4 = piece.get_sides_info()  # [left, bottom, right, top]
    if row == 0 and s4 not in (0, 2): return False  # top edge
    if row == max_row and s2 not in (0, 2): return False  # bottom edge
    if col == 0 and s1 not in (0, 2): return False  # left edge
    if col == max_col and s3 not in (0, 2): return False  # right edge

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
    height = len(grid)
    width = len(grid[0])

    def build_path():
        frame = []
        # ⬅️ Top row (left to right)
        for col in range(width):
            frame.append((0, col))
        # ⬇️ Right column (top to bottom, skip top corner)
        for row in range(1, height):
            frame.append((row, width - 1))
        # ➡️ Bottom row (right to left, skip bottom-right)
        for col in range(width - 2, -1, -1):
            frame.append((height - 1, col))
        # ⬆️ Left column (bottom to top, skip bottom-left and top-left)
        for row in range(height - 2, 0, -1):
            frame.append((row, 0))
        # ⬛ Inner grid (row-major order)
        inside = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
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
    corner_positions = {(0,0),(0, width-1), (height-1, 0), (height-1, width-1)}
    border_positions = {
        *[(0, c) for c in range(1, width-1)],
        *[(height-1, c) for c in range(1, width-1)],
        *[(r, 0) for r in range(1, height - 1)],
        *[(r, width-1) for r in range(1, height - 1)],
    }
    inside_positions = {(r, c) for r in range(1, height-1) for c in range(1, width-1)}

    def backtrack(index, corners_left, borders_left, insides_left, wrongs_left):


        print(f"\n🧩 Progress: {index}/{total_slots} | Corners: {len(corners_left)} | Borders: {len(borders_left)} | Insides: {len(insides_left)} | Wrongs: {len(wrongs_left)}")
        print_grid(grid)

        if index >= total_slots:
            return True

        row, col = full_path[index]

        nb_pieces_inside_th = ((height * width) - (((2 * (height + width)) - 4)))

        is_frame = index < len(full_path) - nb_pieces_inside_th  # 24 - 16 = 8 inside slots

        if is_frame:
            if (row, col) in corner_positions:
                candidates = corners_left
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
        #plot_candidate_scores(scored_candidates, row, col, index)
        print(f"\n📋 Candidates for position ({row}, {col}):")
        for piece, score, rotation in scored_candidates:
            print(f"  - Piece {piece.get_index()} | Score: {score:.3f} | Rotation: {rotation * 90}°")


        for piece, score, rotation in scored_candidates:
            original = piece.clone() if hasattr(piece, "clone") else piece
            for _ in range(rotation):
                rotate_piece(piece)

            if can_place(piece, grid, row, col, corners_left, borders_left, insides_left, wrongs_left):
                print(f"✅ Trying {piece} at ({row}, {col}) with rotation {rotation} and score {score:.2f}")
                grid[row][col] = piece
                visualize_grid(grid)

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




import cv2
import numpy as np

def build_image(solution_indices, pieces, scale_factor=0.5):
    def crop_piece(img):
        if img is None:
            print("⚠️ Warning: Attempted to crop a None image.")
            return np.ones((50, 50, 3), dtype=np.uint8) * 255  # fallback blank image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    rows_images, max_width = [], 0
    for row in solution_indices:
        cropped_imgs, heights = [], []
        for piece in row:
            if piece is not None:
                img = piece.rotate_image_by_rotation()
                img = crop_piece(img)
                img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

                piece_id = f"{piece.get_index()}"
                #cv2.putText(img, piece_id, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


                cropped_imgs.append(img)
                heights.append(img.shape[0])
            else:
                cropped_imgs.append(None)
                heights.append(0)

        row_height = max(heights)
        row_pieces = []
        for img in cropped_imgs:
            if img is not None:
                h, w = img.shape[:2]
                pad_top = (row_height - h) // 2
                pad_bottom = row_height - h - pad_top
                img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
                row_pieces.append(img_padded)
            else:
                row_pieces.append(np.ones((row_height, 1, 3), dtype=np.uint8) * 255)

        row_img = np.hstack(row_pieces)
        rows_images.append(row_img)
        max_width = max(max_width, row_img.shape[1])

    full_height = sum(img.shape[0] for img in rows_images)
    final_img = np.ones((full_height, max_width, 3), dtype=np.uint8) * 255
    y_offset = 0
    for row_img in rows_images:
        final_img[y_offset:y_offset + row_img.shape[0], :row_img.shape[1]] = row_img
        y_offset += row_img.shape[0]

    # Interactive window with rotation
    angle = 0
    window_name = 'Final Puzzle Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        h, w = final_img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions after rotation
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to move the image center to the new canvas center
        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply the rotation with new dimensions
        rotated = cv2.warpAffine(final_img, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        cv2.imshow(window_name, rotated)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            angle += 90
    cv2.destroyAllWindows()

    return final_img


def interactive_view(image):
    angle = 0
    window_name = 'Puzzle Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

        cv2.imshow(window_name, rotated)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC to quit
            break
        elif key == ord('r'):
            angle += 90

    cv2.destroyAllWindows()

def grid_dimensions_from_piece_count(n):
    dims = [(h, n // h) for h in range(1, n + 1) if n % h == 0 and h <= n // h]


    return (sorted(dims, key=lambda x: abs(x[0] - x[1])))[0]

def main():
    # 🧩 Load puzzle
    path = "../res/puzzle.pickle"
    puzzle = p.Puzzle()
    puzzle.load_puzzle(path)
    pieces = puzzle.get_pieces()


    # 👇 Set grid size manually here (rows × columns)
    #rows, cols =  6, 4  # 🔁 Change to (6, 4) if needed
    # ➕ Afficher les dimensions possibles
    total_pieces = len(pieces)
    print(f"\n📦 Nombre total de pièces : {total_pieces}")
    possible_dims = grid_dimensions_from_piece_count(total_pieces)
    print("📐 Dimensions possibles (h x w) :")
    possible_dims_h = possible_dims
    print("dims = ", possible_dims_h)
    rows = possible_dims_h[1]
    cols = possible_dims_h[0]

    print(f"\n📦 Nombre total de pièces : {len(pieces)}")
    print(f"📐 Grid size manually set to: {rows} rows × {cols} columns")

    # 🧠 Classify pieces
    corners, borders, insides, wrongs = classify_pieces(pieces)
    print(f"Corners: {len(corners)} | Borders: {len(borders)} | Insides: {len(insides)} | Wrongs: {len(wrongs)}")

    # 🧩 Build grid
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    print("\n🎯 Starting full puzzle solving...\n")
    if solve_puzzle(grid, corners, borders, insides, wrongs):
        print("✅ Puzzle solved.")
    else:
        print("❌ Puzzle could not be solved.\nTrying rotated dimensions...\n")
        # Try alternative dimensions (flip)
        grid = [[None for _ in range(rows)] for _ in range(cols)]
        if solve_puzzle(grid, corners, borders, insides, wrongs):
            print("✅ Puzzle solved with fallback grid.")
        else:
            print("❌ Puzzle could not be solved with fallback either.")

    print_grid(grid)

    # 🖼️ Build final image with piece labels
    final_image = build_image(grid, pieces)
    cv2.imshow("Solved Puzzle", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
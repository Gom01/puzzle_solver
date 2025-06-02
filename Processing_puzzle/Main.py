import numpy as np
import Puzzle as puzzle
import cv2

from Processing_puzzle.parsing.color_analysis_v2 import find_color
from Processing_puzzle.parsing.corners import find_corners
from Processing_puzzle.parsing.parse_using_backgrounds import process_puzzle_images
from Processing_puzzle.parsing.sides_analysis import sides_information
from Processing_puzzle.parsing.sides_finder import find_sides
from Processing_puzzle.parsing.straight_piece import straighten_piece

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def main():
    paths = [
        '../images/pictures/puzzleB/white.jpg',
        '../images/pictures/puzzleB/blue.jpg',
        '../images/pictures/puzzleB/red.jpg',
        '../images/pictures/puzzleB/green.jpg'
    ]

    print("⚠️  Ensure no pieces are touching and use background of different color!")
    print("⚠️  Be sure to control each pieces !")

    myPuzzle = puzzle.Puzzle()
    process_puzzle_images(paths, myPuzzle, False)
    print("📷 Step 1: Puzzle images processed.")
    find_corners(myPuzzle, False)
    print("📐 Step 2: Corners detected.")
    find_sides(myPuzzle, False)
    print("🧩 Step 3: Sides extracted.")
    find_color(myPuzzle, False)
    print("🎨 Step 4: Color features analyzed.")
    sides_information(myPuzzle, False)
    print("🔍 Step 5: Side types (tabs/blanks/flats) identified.")
    straighten_piece(myPuzzle, False)
    print("📏 Step 6: Pieces straightened and aligned.")

    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):
        sides = piece.get_sides()
        img = piece.get_color_image().copy()
        corners = piece.get_corners()

        print(f"Side info : {piece.get_sides_info()}")
        print(f"Corners   : {corners}")
        for i in range(4):
            print(f"Contours {i+1}: {sides[i].get_side_contour()}")

        if corners[0] == [2, 2]:
            print(f"⚠️ Invalid piece {idx}")
            continue

        h, w = img.shape[:2]
        max_display_dim = 800
        if max(h, w) > max_display_dim:
            scale = max_display_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            scale_coords = lambda pt: (int(pt[0] * scale), int(pt[1] * scale))
        else:
            scale_coords = lambda pt: (int(pt[0]), int(pt[1]))

        # --- Draw side labels using fixed corner mapping ---
        # S0: C3→C0 (left), S1: C2→C3 (bottom), S2: C1→C2 (right), S3: C0→C1 (top)
        side_corner_pairs = [(3, 0), (2, 3), (1, 2), (0, 1)]
        for i, (a, b) in enumerate(side_corner_pairs):
            pt1 = corners[a]
            pt2 = corners[b]
            mid = scale_coords(midpoint(pt1, pt2))
            label = f"{sides[i].get_side_info()}"
            cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        # --- Draw corners ---
        for j, (x, y) in enumerate(corners):
            px, py = scale_coords((x, y))
            cv2.circle(img, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(img, f"{j}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # --- Show the piece ---
        window_name = f'Piece {idx}'
        cv2.imshow(window_name, img)

        print(f"\nInspecting Piece {idx}. Press:")
        print("  [g] → Good")
        print("  [b] → Bad")
        print("  [Esc] or [Enter] → Continue to next piece")

        while True:
            key = cv2.waitKey(0)
            if key == ord('b'):
                piece.is_bad = True
                print(f"Marked Piece {idx} as BAD ✅")
                break
            elif key == ord('g'):
                print(f"Marked Piece {idx} as GOOD 👍")
                break
            elif key in [27, 13]:
                break

        cv2.destroyWindow(window_name)

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("✅ Puzzle saved successfully.")

if __name__ == "__main__":
    main()

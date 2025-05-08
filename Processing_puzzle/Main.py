import Puzzle as puzzle
import cv2

from Processing_puzzle.parsing.straight_piece import straighten_piece
from Processing_puzzle.parsing.color_analysis_v2 import find_color
from Processing_puzzle.parsing.corners import find_corners
from Processing_puzzle.parsing.parse import parse_image
from Processing_puzzle.parsing.sides_analysis import sides_information
from Processing_puzzle.parsing.sides_finder import find_sides


def main():
    image_path = '../images/p1_b/puzzle_24_bis.jpg' # Fonctionne
    #image_path = '../images/p1_b/Natel.Black1.jpg' # marche pas
    #image_path = '../images/pictures/puzzle1.jpg' #  marche pas
    #image_path = '../images/pictures/puzzle2.jpg' # marche pas sur la 2ème étape sur le Matching
    #image_path = '../images/pictures/puzzle49.jpg'


    # Important note for image acquisition
    print("⚠️  Ensure no pieces are touching and a black background is used.")

    myPuzzle = puzzle.Puzzle()

    parse_image(image_path, myPuzzle)
    find_corners(myPuzzle)
    find_sides(myPuzzle)
    find_color(myPuzzle, False)
    sides_information(myPuzzle)
    straighten_piece(myPuzzle)

    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):
        infos = piece.get_sides_info()
        sides = piece.get_sides()
        img = piece.get_color_image().copy()

        # Annotate side indices
        for i, side in enumerate(sides):
            contour = side.get_side_contour()
            midpoint = contour[len(contour) // 2]
            cv2.putText(img, str(f"{i}"), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 5, cv2.LINE_AA)

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
            elif key == 27 or key == 13:  # Esc or Enter
                break

        cv2.destroyWindow(window_name)

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("✅ Puzzle saved successfully.")


if __name__ == "__main__":
    main()

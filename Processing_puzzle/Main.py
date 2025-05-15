import Puzzle as puzzle
import cv2

from Processing_puzzle.parsing.color_analysis_v2 import find_color
from Processing_puzzle.parsing.corners import find_corners
from Processing_puzzle.parsing.parse_using_backgrounds import process_puzzle_images
from Processing_puzzle.parsing.sides_analysis import sides_information
from Processing_puzzle.parsing.sides_finder import find_sides
from Processing_puzzle.parsing.straight_piece import straighten_piece


def main():
    paths = [
        '../images/pictures/puzzleB/white.jpg',
        '../images/pictures/puzzleB/red.jpg',
        '../images/pictures/puzzleB/green.jpg',
        '../images/pictures/puzzleB/blue.jpg'
    ]


    # Important note for image acquisition
    print("⚠️  Ensure no pieces are touching and use background of different color!")

    myPuzzle = puzzle.Puzzle()
    process_puzzle_images(paths, myPuzzle, False)
    find_corners(myPuzzle, False)
    find_sides(myPuzzle, False)
    find_color(myPuzzle, False)
    sides_information(myPuzzle, False)
    straighten_piece(myPuzzle)

    pieces = myPuzzle.get_pieces()

    for idx, piece in enumerate(pieces):

        infos = piece.get_sides_info()
        sides = piece.get_sides()
        img = piece.get_color_image().copy()
        corners = piece.get_corners()





        # Annotate side indices
        for i, side in enumerate(sides):
            contour = side.get_side_contour()
            midpoint = contour[len(contour) // 2]
            cv2.putText(img, str(f"{side.get_side_info()}"), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        print(piece.get_sides_info())
        window_name = f'Piece {idx+1}'
        cv2.imshow(window_name, img)


        print(f"\nInspecting Piece {idx+1}. Press:")
        print("  [g] → Good")
        print("  [b] → Bad")
        print("  [Esc] or [Enter] → Continue to next piece")

        while True:
            key = cv2.waitKey(0)
            if key == ord('b'):
                piece.is_bad = True
                print(f"Marked Piece {idx+1} as BAD ✅")
                break
            elif key == ord('g'):
                print(f"Marked Piece {idx+1} as GOOD 👍")
                break
            elif key == 27 or key == 13:  # Esc or Enter
                break

        cv2.destroyWindow(window_name)

    myPuzzle.save_puzzle('../Processing_puzzle/res/puzzle.pickle')
    print("✅ Puzzle saved successfully.")


if __name__ == "__main__":
    main()

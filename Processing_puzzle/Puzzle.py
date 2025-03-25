import pickle

class Puzzle:
    def __init__(self):
        self.pieces = []

    def add_piece(self, piece):
        self.pieces.append(piece)

    def save_puzzle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.pieces, f)

    def load_puzzle(self, filename):
        with open(filename, 'rb') as f:
            self.pieces = pickle.load(f)

    def get_pieces(self):
        return self.pieces

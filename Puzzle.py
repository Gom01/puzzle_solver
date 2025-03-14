import pickle

class Puzzle:
    def __init__(self):
        """Initialise le puzzle avec une liste vide de pièces."""
        self.pieces = []

    def add_pieceset(self, piece):
        """
        Ajoute une pièce au puzzle.

        :param piece: Instance de la classe Piece_Puzzle.
        """
        self.pieces.append(piece)

    def save_puzzle(self, filename):
        """
        Sauvegarde l'état du puzzle dans un fichier pickle.

        :param filename: Le nom du fichier où enregistrer l'état du puzzle.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.pieces, f)

    def load_puzzle(self, filename):
        """
        Charge un puzzle depuis un fichier pickle.

        :param filename: Le nom du fichier à partir duquel charger le puzzle.
        """
        with open(filename, 'rb') as f:
            self.pieces = pickle.load(f)

    def get_pieces(self):
        """Retourne toutes les pièces du puzzle."""
        return self.pieces

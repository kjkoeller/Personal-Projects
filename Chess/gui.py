import tkinter as tk
from chess_bitboard import ChessBitboard  # Make sure this import is correct


class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game")

        self.chess = ChessBitboard()  # Initialize ChessBitboard
        self.selected_pos = None
        self.selected_piece = None
        self.turn = 'white'

        self.board_buttons = [[None for _ in range(8)] for _ in range(8)]
        self.create_board()

    def create_board(self):
        """Create the 8x8 board with buttons and labels."""
        for col in range(8):
            tk.Label(self.root, text=chr(ord('A') + col), width=4, height=2, bg='lightgray').grid(row=0, column=col + 1)

        for row in range(8):
            tk.Label(self.root, text=str(8 - row), width=4, height=2, bg='lightgray').grid(row=row + 1, column=0)

        for row in range(8):
            for col in range(8):
                color = 'white' if (row + col) % 2 == 0 else 'gray'
                button = tk.Button(self.root, width=4, height=2, bg=color,
                                   command=lambda r=row, c=col: self.on_click(r, c))
                button.grid(row=row + 1, column=col + 1)
                self.board_buttons[row][col] = button

        self.update_board()

    def update_board(self):
        """Update the board GUI with the current state."""
        board_state = self.chess.get_board()
        for r in range(8):
            for c in range(8):
                piece = board_state[r][c]
                color = 'white' if piece.isupper() else 'black'
                self.board_buttons[r][c].config(text=piece, bg='white' if color == 'white' else 'gray')

    def on_click(self, row, col):
        """Handle click events on the board."""
        if self.selected_pos is None:
            piece = self.chess.get_board()[row][col]
            if piece != '.':
                if (self.turn == 'white' and piece.isupper()) or (self.turn == 'black' and piece.islower()):
                    self.selected_pos = (row, col)
                    self.selected_piece = piece
                    self.highlight_moves(row, col)
                    print(f"Selected piece: {piece} at {self.selected_pos}")
        else:
            if self.is_legal_move(self.selected_pos, (row, col), self.selected_piece):
                print(f"Moving piece from {self.selected_pos} to {(row, col)}")
                self.chess.move_piece(self.selected_pos, (row, col))
                self.selected_pos = None
                self.selected_piece = None
                self.turn = 'black' if self.turn == 'white' else 'white'
                self.update_board()
            else:
                print(f"Illegal move from {self.selected_pos} to {(row, col)}")
                self.selected_pos = None
                self.selected_piece = None
                self.update_board()

    def highlight_moves(self, row, col):
        """Highlight possible moves for the selected piece."""
        # For debugging, we could add some indication here
        # Example: highlight possible moves (not implemented here)
        print(f"Highlighting moves for {self.chess.get_board()[row][col]} at {(row, col)}")

    def is_legal_move(self, from_pos, to_pos, piece):
        """Check if a move is legal."""
        return self.chess.is_legal_move(from_pos, to_pos, piece)


if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()
class ChessBitboard:
    def __init__(self):
        self.init_bitboards()
        self.last_opponent_move = None  # Initialize last_opponent_move
        self.move_history = []

    def init_bitboards(self):
        self.white_pawns = 0x00FF000000000000
        self.black_pawns = 0x000000000000FF00
        self.white_rooks = 0x8100000000000000
        self.black_rooks = 0x0000000000000081
        self.white_knights = 0x4200000000000000
        self.black_knights = 0x0000000000000042
        self.white_bishops = 0x2400000000000000
        self.black_bishops = 0x0000000000000024
        self.white_queens = 0x1000000000000000
        self.black_queens = 0x0000000000000010
        self.white_king = 0x0800000000000000
        self.black_king = 0x0000000000000008

        self.update_occupied_bitboards()

    def update_occupied_bitboards(self):
        self.occupied_white = (
                self.white_pawns | self.white_rooks | self.white_knights |
                self.white_bishops | self.white_queens | self.white_king
        )
        self.occupied_black = (
                self.black_pawns | self.black_rooks | self.black_knights |
                self.black_bishops | self.black_queens | self.black_king
        )
        self.all_occupied = self.occupied_white | self.occupied_black

    def get_board(self):
        """Return the current board state as a 2D list."""
        board = [['.' for _ in range(8)] for _ in range(8)]

        def bitboard_to_pieces(bitboard, piece_char):
            for i in range(64):
                if bitboard & (1 << i):
                    row, col = divmod(i, 8)
                    board[row][col] = piece_char

        bitboard_to_pieces(self.white_pawns, 'P')
        bitboard_to_pieces(self.black_pawns, 'p')
        bitboard_to_pieces(self.white_rooks, 'R')
        bitboard_to_pieces(self.black_rooks, 'r')
        bitboard_to_pieces(self.white_knights, 'N')
        bitboard_to_pieces(self.black_knights, 'n')
        bitboard_to_pieces(self.white_bishops, 'B')
        bitboard_to_pieces(self.black_bishops, 'b')
        bitboard_to_pieces(self.white_queens, 'Q')
        bitboard_to_pieces(self.black_queens, 'q')
        bitboard_to_pieces(self.white_king, 'K')
        bitboard_to_pieces(self.black_king, 'k')

        return board

    def move_piece(self, from_pos, to_pos):
        """Move a piece from from_pos to to_pos."""
        piece = self.get_piece_at(from_pos)
        if piece is None:
            return
        
        if piece.lower() == 'p':
            if self.en_passant_capture(from_pos, to_pos, piece):
                if piece.islower():
                    captured_pos = (to_pos[0]-1, to_pos[1])
                else:
                    captured_pos = (to_pos[0]+1, to_pos[1])
                self.remove_piece(captured_pos)
        
        original_piece = self.get_piece_at(to_pos)
        
        if self.is_occupied_by_opponent(to_pos, piece):
            # remove opponent's piece
            self.remove_piece(to_pos)
        
        # Remove the piece from the board
        self.remove_piece(from_pos)

        # Place the piece at the new position
        self.place_piece(to_pos, piece)
        
        # checks whether the move leaves the king in check
        color = 'white' if piece.isupper() else 'black'
        if self.is_in_check(color):
            # revert the move
            self.remove_piece(to_pos)
            self.place_piece(from_pos, piece)
            if original_piece is not None:
                self.place_piece(to_pos, original_piece)
            print("\n\nMove is illegal: King is in Check!\n\n")
            return False # invalid move, king would be in check

        # Update last opponent move
        self.last_opponent_move = (from_pos, to_pos)

        # Update bitboards
        self.update_occupied_bitboards()
        
        return True # move is valid
    
    def revert_move(self, from_pos, to_pos):
        if not self.move_history:
            return False
        
        last_board, last_from_pos, last_to_pos = self.move_history.pop()
        
        self.board = last_board
        
        return True

    def is_within_bounds(self, pos):
        """Check if a position is within the board bounds."""
        row, col = pos
        return 0 <= row < 8 and 0 <= col < 8

    def is_empty(self, pos):
        """Check if a position is empty."""
        row, col = pos
        return not self.all_occupied & (1 << (row * 8 + col))

    def is_occupied_by_opponent(self, pos, piece):
        """Check if a position is occupied by an opponent's piece."""
        row, col = pos
        if piece.isupper():
            return self.black_pawns & (1 << (row * 8 + col)) or \
                self.black_rooks & (1 << (row * 8 + col)) or \
                self.black_knights & (1 << (row * 8 + col)) or \
                self.black_bishops & (1 << (row * 8 + col)) or \
                self.black_queens & (1 << (row * 8 + col)) or \
                self.black_king & (1 << (row * 8 + col))
        else:
            return self.white_pawns & (1 << (row * 8 + col)) or \
                self.white_rooks & (1 << (row * 8 + col)) or \
                self.white_knights & (1 << (row * 8 + col)) or \
                self.white_bishops & (1 << (row * 8 + col)) or \
                self.white_queens & (1 << (row * 8 + col)) or \
                self.white_king & (1 << (row * 8 + col))

    def is_legal_move(self, from_pos, to_pos, piece):
        if not self.is_within_bounds(from_pos) or not self.is_within_bounds(to_pos):
            return False
        
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        if piece.upper() == 'P':
            # Pawn move validation
            direction = -1 if piece.isupper() else 1 # white moves up (-1) and black moves down (1)
            start_row = 6 if piece.isupper() else 1
            
            if to_pos[1] == from_pos[1]:
                if self.get_piece_at(to_pos) is not None:
                    return False # can't move forward if blocked
                if from_pos[0] + direction == to_pos[0]:
                    return True # one move forward
                if from_pos[0] == start_row and from_pos[0] + 2 * direction == to_pos[0]:
                    # two square movements
                    intermediate_pos = (from_pos[0] + direction, from_pos[1])
                    if self.get_piece_at(intermediate_pos) is None:
                        return True
            
            if abs(to_pos[1] - from_pos[1]) == 1 and from_pos[0] + direction == to_pos[0]:
                target_piece = self.get_piece_at(to_pos)
                if target_piece is not None and self.is_occupied_by_opponent(to_pos, piece):
                    return True # standard diagonal capture
                
                if self.en_passant_capture(from_pos, to_pos, piece):
                    return True # en passant capture
            
            return False
                
        
        elif piece.upper() == 'N':
            # Knight move validation
            row_diff = abs(from_row - to_row)
            col_diff = abs(from_col - to_col)
            
            if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
                if self.is_empty(to_pos) or self.is_occupied_by_opponent(to_pos, piece):
                    return True

        elif piece.upper() == 'R':
            # Rook move validation
            if from_row == to_row or from_col == to_col:
                return self.is_path_clear(from_pos, to_pos)

        elif piece.upper() == 'B':
            # Bishop move validation
            if abs(from_row - to_row) == abs(from_col - to_col):
                return self.is_path_clear(from_pos, to_pos)

        elif piece.upper() == 'Q':
            # Queen move validation
            if from_row == to_row or from_col == to_col or \
                    abs(from_row - to_row) == abs(from_col - to_col):
                return self.is_path_clear(from_pos, to_pos)

        elif piece.upper() == 'K':
            # King move validation
            if abs(from_row - to_row) <= 1 and abs(from_col - to_col) <= 1:
                return True

        return False

    def is_path_clear(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        if from_row == to_row:  # Horizontal move
            step = 1 if from_col < to_col else -1
            for col in range(from_col + step, to_col, step):
                if not self.is_empty((from_row, col)):
                    return False
        elif from_col == to_col:  # Vertical move
            step = 1 if from_row < to_row else -1
            for row in range(from_row + step, to_row, step):
                if not self.is_empty((row, from_col)):
                    return False
        elif abs(from_row - to_row) == abs(from_col - to_col):  # Diagonal move
            row_step = 1 if from_row < to_row else -1
            col_step = 1 if from_col < to_col else -1
            row, col = from_row + row_step, from_col + col_step
            while row != to_row:
                if not self.is_empty((row, col)):
                    return False
                row += row_step
                col += col_step
        return True

    def is_checkmate(self, color):
        """Check if the given color is in checkmate."""
        if not self.is_in_check(color):
            return False

        pieces = self.get_pieces(color)
        for piece in pieces:
            from_pos = self.get_piece_position(piece)
            for move in self.get_possible_moves(from_pos, piece):
                if not self.is_in_check_after_move(from_pos, move, color):
                    return False
        return True

    def is_in_check_after_move(self, from_pos, to_pos, color):
        """Check if the king is in check after a move."""
        # Save the board state
        original_piece = self.get_piece_at(from_pos)
        target_piece = self.get_piece_at(to_pos)

        # Perform the move
        self.move_piece(from_pos, to_pos)

        # Check if the king is in check
        # king_pos = self.get_king_position(color)
        in_check = self.is_in_check(color)

        # Revert the move
        self.remove_piece(to_pos)
        self.place_piece(from_pos, original_piece)
        if target_piece != '.':
            self.place_piece(to_pos, target_piece)

        return in_check
    
    def is_in_check(self, color):
        king_pos = self.find_king(color)
        if not king_pos:
            return False
        
        opponent_color = 'black' if color == 'white' else 'white'
        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at((row, col))
                if piece and self.get_piece_color(piece) == opponent_color:
                    if self.is_legal_move((row, col), king_pos, piece):
                        return True # the king is in check
        return False
    
    def find_king(self, color):
        king_char = 'K' if color=='white' else 'k'
        for row in range(8):
            for col in range(8):
                if self.get_piece_at((row, col)) == king_char:
                    return (row, col)
        return None
    
    def get_piece_color(self, piece):
        return 'white' if piece.isupper() else 'black'
        

    def get_pieces(self, color):
        """Get all pieces of the given color."""
        if color == 'white':
            return self.white_pawns | self.white_rooks | self.white_knights | \
                self.white_bishops | self.white_queens | self.white_king
        else:
            return self.black_pawns | self.black_rooks | self.black_knights | \
                self.black_bishops | self.black_queens | self.black_king

    def get_piece_at(self, pos):
        """Get the piece at the given position."""
        row, col = pos
        index = row * 8 + col

        if self.white_pawns & (1 << index):
            return 'P'
        if self.black_pawns & (1 << index):
            return 'p'
        if self.white_rooks & (1 << index):
            return 'R'
        if self.black_rooks & (1 << index):
            return 'r'
        if self.white_knights & (1 << index):
            return 'N'
        if self.black_knights & (1 << index):
            return 'n'
        if self.white_bishops & (1 << index):
            return 'B'
        if self.black_bishops & (1 << index):
            return 'b'
        if self.white_queens & (1 << index):
            return 'Q'
        if self.black_queens & (1 << index):
            return 'q'
        if self.white_king & (1 << index):
            return 'K'
        if self.black_king & (1 << index):
            return 'k'

        return '.'

    def can_castle(self, color, side):
        """Check if castling is possible for the given color and side."""
        if color == 'white':
            king_pos = (7, 4)
            if side == 'kingside':
                rook_pos = (7, 7)
                rook_bitboard = self.white_rooks
                path = [(7, 5), (7, 6)]
            else:
                rook_pos = (7, 0)
                rook_bitboard = self.white_rooks
                path = [(7, 1), (7, 2), (7, 3)]
        else:
            king_pos = (0, 4)
            if side == 'kingside':
                rook_pos = (0, 7)
                rook_bitboard = self.black_rooks
                path = [(0, 5), (0, 6)]
            else:
                rook_pos = (0, 0)
                rook_bitboard = self.black_rooks
                path = [(0, 1), (0, 2), (0, 3)]

        if not (self.is_legal_rook_move(rook_pos, king_pos, 'R') and
                self.is_legal_king_move(king_pos, path[-1])):
            return False

        if not (self.is_legal_rook_move(rook_pos, path[0], 'R') and
                self.is_legal_king_move(king_pos, path[0])):
            return False

        if self.is_in_check(color):
            return False

        for pos in path:
            if self.is_occupied(pos) or self.is_in_check_after_move(king_pos, pos, color):
                return False

        return True

    def en_passant_capture(self, from_pos, to_pos, piece):
        """Check if an en passant capture is possible."""
        if piece.lower() != 'p':
            return False
        
        start_row, start_col = from_pos
        end_row, end_col = to_pos

        if not self.last_opponent_move:
            print("No last opponent move")
            return False

        opp_from_pos, opp_to_pos = self.last_opponent_move
        opponent_piece = self.get_piece_at(opp_to_pos)
        
        if opponent_piece.lower() == 'p':
            if self.is_opponent_pawn_two_square_move(opp_from_pos, opp_to_pos):
                if piece.isupper(): # white pawn
                    return start_row == 3 and end_row == 2 and abs(start_col - end_col) == 1 and opp_to_pos == (start_row, start_col)
                else: # black pawn
                    return start_row == 4 and end_row == 5 and abs(start_col - end_col) == 1 and opp_to_pos == (start_row, start_col)
        
        return False
    
    def is_opponent_pawn_two_square_move(self, opp_from_pos, opp_to_pos):
        """"Check to make sure the last opponent pawn move was two squares"""
        start_row, _ = opp_from_pos
        end_row, _ = opp_to_pos
        
        return (start_row == 6 and end_row == 4) or (start_row == 1 and end_row == 3)
        

    def promote_pawn(self, position, color, promotion_piece):
        """Promote a pawn to a new piece."""
        row, col = position
        if color == 'white':
            self.white_pawns ^= (1 << (7 - row) * 8 + col)
            if promotion_piece == 'Q':
                self.white_queens |= (1 << (7 - row) * 8 + col)
            elif promotion_piece == 'R':
                self.white_rooks |= (1 << (7 - row) * 8 + col)
            elif promotion_piece == 'B':
                self.white_bishops |= (1 << (7 - row) * 8 + col)
            elif promotion_piece == 'N':
                self.white_knights |= (1 << (7 - row) * 8 + col)
        else:
            self.black_pawns ^= (1 << row * 8 + col)
            if promotion_piece == 'q':
                self.black_queens |= (1 << row * 8 + col)
            elif promotion_piece == 'r':
                self.black_rooks |= (1 << row * 8 + col)
            elif promotion_piece == 'b':
                self.black_bishops |= (1 << row * 8 + col)
            elif promotion_piece == 'n':
                self.black_knights |= (1 << row * 8 + col)

        self.update_occupied_bitboards()

    def get_piece_at(self, pos):
        """Get the piece at the given position."""
        row, col = pos
        index = row * 8 + col

        if self.white_pawns & (1 << index):
            return 'P'
        elif self.black_pawns & (1 << index):
            return 'p'
        elif self.white_rooks & (1 << index):
            return 'R'
        elif self.black_rooks & (1 << index):
            return 'r'
        elif self.white_knights & (1 << index):
            return 'N'
        elif self.black_knights & (1 << index):
            return 'n'
        elif self.white_bishops & (1 << index):
            return 'B'
        elif self.black_bishops & (1 << index):
            return 'b'
        elif self.white_queens & (1 << index):
            return 'Q'
        elif self.black_queens & (1 << index):
            return 'q'
        elif self.white_king & (1 << index):
            return 'K'
        elif self.black_king & (1 << index):
            return 'k'
        return None

    def remove_piece(self, pos):
        """Remove a piece from the given position."""
        row, col = pos
        index = row * 8 + col

        self.white_pawns &= ~(1 << index)
        self.black_pawns &= ~(1 << index)
        self.white_rooks &= ~(1 << index)
        self.black_rooks &= ~(1 << index)
        self.white_knights &= ~(1 << index)
        self.black_knights &= ~(1 << index)
        self.white_bishops &= ~(1 << index)
        self.black_bishops &= ~(1 << index)
        self.white_queens &= ~(1 << index)
        self.black_queens &= ~(1 << index)
        self.white_king &= ~(1 << index)
        self.black_king &= ~(1 << index)

        self.update_occupied_bitboards()

    def place_piece(self, pos, piece):
        """Place a piece at the given position."""
        row, col = pos
        index = row * 8 + col

        if piece == 'P':
            self.white_pawns |= (1 << index)
        elif piece == 'p':
            self.black_pawns |= (1 << index)
        elif piece == 'R':
            self.white_rooks |= (1 << index)
        elif piece == 'r':
            self.black_rooks |= (1 << index)
        elif piece == 'N':
            self.white_knights |= (1 << index)
        elif piece == 'n':
            self.black_knights |= (1 << index)
        elif piece == 'B':
            self.white_bishops |= (1 << index)
        elif piece == 'b':
            self.black_bishops |= (1 << index)
        elif piece == 'Q':
            self.white_queens |= (1 << index)
        elif piece == 'q':
            self.black_queens |= (1 << index)
        elif piece == 'K':
            self.white_king |= (1 << index)
        elif piece == 'k':
            self.black_king |= (1 << index)

        self.update_occupied_bitboards()

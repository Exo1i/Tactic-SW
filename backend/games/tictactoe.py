from typing import Optional, Literal

class TicTacToe:
    def __init__(self):
        self.board: list[list[Optional[str]]] = [[None for _ in range(3)] for _ in range(3)]
        self.current_player: Literal["X", "O"] = "X"
        self.winner: Optional[str] = None
        self.moves: int = 0

    def reset(self) -> None:
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.winner = None
        self.moves = 0

    def make_move(self, row: int, col: int) -> bool:
        if self.board[row][col] is not None or self.winner is not None:
            return False
        self.board[row][col] = self.current_player
        self.moves += 1
        if self.check_winner(row, col):
            self.winner = self.current_player
        elif self.moves == 9:
            self.winner = "Draw"
        else:
            self.current_player = "O" if self.current_player == "X" else "X"
        return True

    def check_winner(self, row: int, col: int) -> bool:
        b = self.board
        p = self.current_player
        return (
            all(b[row][c] == p for c in range(3)) or
            all(b[r][col] == p for r in range(3)) or
            (row == col and all(b[i][i] == p for i in range(3))) or
            (row + col == 2 and all(b[i][2 - i] == p for i in range(3)))
        )

    def get_state(self) -> dict:
        return {
            "board": self.board,
            "current_player": self.current_player,
            "winner": self.winner,
            "moves": self.moves,
        }

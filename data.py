import random
import pandas as pd

# Constants for the game
BOARD_DIM = 13  # Board size
NUM_GAMES = 10000  # Number of games to simulate
EMPTY = " "  # Empty cell representation
PLAYER_X = "X"
PLAYER_O = "O"
DIRECTIONS = [(0, 1), (1, 0), (1, -1)]  # Hexagonal neighbors

class HexGame:
    def __init__(self, board_dim):
        self.board_dim = board_dim
        self.board = [[EMPTY for _ in range(board_dim)] for _ in range(board_dim)]
        self.open_positions = [(i, j) for i in range(board_dim) for j in range(board_dim)]

    def is_valid_position(self, i, j):
        return 0 <= i < self.board_dim and 0 <= j < self.board_dim

    def neighbors(self, i, j):
        """Get all neighbors of a cell."""
        for di, dj in DIRECTIONS + [(-di, -dj) for di, dj in DIRECTIONS]:
            ni, nj = i + di, j + dj
            if self.is_valid_position(ni, nj):
                yield ni, nj

    def place_piece(self, i, j, player):
        self.board[i][j] = player
        self.open_positions.remove((i, j))

    def check_connection(self, i, j, player, visited):
        """DFS to check if the player has a connected path."""
        if (player == PLAYER_X and j == self.board_dim - 1) or (player == PLAYER_O and i == self.board_dim - 1):
            return True
        visited.add((i, j))
        for ni, nj in self.neighbors(i, j):
            if (ni, nj) not in visited and self.board[ni][nj] == player:
                if self.check_connection(ni, nj, player, visited):
                    return True
        return False

    def has_winner(self, player):
        """Check if the player has a winning path."""
        visited = set()
        if player == PLAYER_X:
            for i in range(self.board_dim):
                if self.board[i][0] == player and self.check_connection(i, 0, player, visited):
                    return True
        elif player == PLAYER_O:
            for j in range(self.board_dim):
                if self.board[0][j] == player and self.check_connection(0, j, player, visited):
                    return True
        return False

    def heuristic_move(self, player):
        """Choose a move based on heuristics."""
        # Prioritize connecting to the player's existing pieces
        for i, j in self.open_positions:
            for ni, nj in self.neighbors(i, j):
                if self.board[ni][nj] == player:
                    return i, j

        # Block opponent's critical moves
        opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
        for i, j in self.open_positions:
            for ni, nj in self.neighbors(i, j):
                if self.board[ni][nj] == opponent:
                    return i, j

        # Default: choose a random move
        return random.choice(self.open_positions)

    def board_to_string(self):
        """Convert the board to a single string representation."""
        return "".join("".join(row) for row in self.board)

    def play_game(self):
        """Simulate a game until there is a winner or the board is full.
        Returns (winner, list_of_states) where list_of_states are the board states after each move.
        """
        states = []
        player = random.choice([PLAYER_X, PLAYER_O])  # Random starting player
        while self.open_positions:
            move = self.heuristic_move(player)
            self.place_piece(*move, player)
            # Record the current state after the move
            states.append(self.board_to_string())
            if self.has_winner(player):
                return player, states
            player = PLAYER_O if player == PLAYER_X else PLAYER_X
        return None, states  # Draw scenario

def generate_hex_datasets(board_dim, num_games,
                          final_file="10_hex_13x13.csv",
                          two_moves_file="10_hex_13x13_2moves.csv",
                          five_moves_file="10_hex_13x13_5moves.csv"):
    final_records = []
    two_moves_records = []
    five_moves_records = []

    for _ in range(num_games):
        game = HexGame(board_dim)
        winner, states = game.play_game()

        if winner is not None:
            # final state: always available as states[-1]
            final_board = states[-1]
            winner_label = "1" if winner == PLAYER_X else "0"
            final_records.append((final_board, winner_label))

            # 2 moves before final: states[-3]
            # This requires at least 3 moves before the final (i.e. len(states) >= 3)
            if len(states) >= 3:
                two_moves_board = states[-3]
                two_moves_records.append((two_moves_board, winner_label))

            # 5 moves before final: states[-6]
            # This requires at least 6 moves before the final (i.e. len(states) >= 6)
            if len(states) >= 6:
                five_moves_board = states[-6]
                five_moves_records.append((five_moves_board, winner_label))

    # Save final dataset
    pd.DataFrame(final_records, columns=["board", "winner"]).to_csv(final_file, index=False)
    print(f"Saved {len(final_records)} final state entries to {final_file}.")

    # Save two moves dataset
    pd.DataFrame(two_moves_records, columns=["board", "winner"]).to_csv(two_moves_file, index=False)
    print(f"Saved {len(two_moves_records)} 2-moves-before entries to {two_moves_file}.")

    # Save five moves dataset
    pd.DataFrame(five_moves_records, columns=["board", "winner"]).to_csv(five_moves_file, index=False)
    print(f"Saved {len(five_moves_records)} 5-moves-before entries to {five_moves_file}.")

if __name__ == "__main__":
    generate_hex_datasets(BOARD_DIM, NUM_GAMES)

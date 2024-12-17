import numpy as np
import pandas as pd
import os
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

#######################
# Data Loading and Preprocessing
#######################

def load_hex_dataset(csv_path):
    """
    Load Hex board states and winner outcomes from a CSV file.
    Expects columns: cell0, cell1, ..., cell48, winner
    Each cell: 1 (Player 1), -1 (Player 2), 0 (empty)
    winner: 1 (Player 1), -1 (Player 2)
    """
    df = pd.read_csv(csv_path)
    board_columns = [col for col in df.columns if col.startswith('cell')]
    boards = df[board_columns].values  # shape (N, 49) for a 7x7 board
    winners = df['winner'].values
    return boards, winners

def create_partial_states(boards, moves_before_end):
    """
    Given final boards and a number of moves (n_moves), return a copy of boards
    with the last n_moves placed stones removed (set to 0).
    We assume that 'last moves' correspond to the last non-zero cells in the board array.
    """
    partial_boards = boards.copy()
    for i, board in enumerate(boards):
        non_zero_indices = np.flatnonzero(board)
        if len(non_zero_indices) >= moves_before_end:
            # Remove the last 'moves_before_end' stones placed
            indices_to_clear = non_zero_indices[-moves_before_end:]
            partial_boards[i, indices_to_clear] = 0
    return partial_boards

def convert_label(winner):
    # Convert {1, -1} to {1, 0}
    return 1 if winner == 1 else 0

def hex_neighbors(r, c, board_size=7):
    candidates = [
       (r-1, c), (r+1, c),
       (r, c-1), (r, c+1),
       (r-1, c+1), (r+1, c-1)
    ]
    return [(rr, cc) for (rr, cc) in candidates if 0 <= rr < board_size and 0 <= cc < board_size]

#######################
# Main Code
#######################
if __name__ == "__main__":
    csv_path = "reduced_hex_games.csv"

    # Load final boards and winners
    boards, winners = load_hex_dataset(csv_path)

    # Create datasets for final, 2 moves before, 5 moves before
    board_states = {
        "final": boards,
        "2_moves_before": create_partial_states(boards, 2),
        "5_moves_before": create_partial_states(boards, 5)
    }

    # We will process each scenario similarly
    # For demonstration, let's just use a small subset of the data
    # to keep runtime reasonable.
    N_total = boards.shape[0]
    N_train = min(N_total, 10000)  # Use up to 1000 samples

    board_size = 7
    num_nodes = board_size * board_size
    symbols = ['P1', 'P2', 'Empty']  # node properties

    # Hyperparameters inspired by the MNIST example
    number_of_clauses = 2000
    T = 5000
    s = 10.0
    number_of_state_bits = 8
    epochs = 20

    # Loop through each scenario
    for scenario_name, scenario_boards in board_states.items():
        print(f"\n### Scenario: {scenario_name} ###")

        # Slice data
        scenario_boards = scenario_boards[:N_train]
        scenario_winners = winners[:N_train]
        Y = np.array([convert_label(w) for w in scenario_winners], dtype=np.uint32)

        # Initialize Graphs
        graphs = Graphs(
            N_train,
            symbols=symbols,
            hypervector_size=128,
            hypervector_bits=2
        )

        # Set number of nodes per graph
        for g_id in range(N_train):
            graphs.set_number_of_graph_nodes(g_id, num_nodes)
        graphs.prepare_node_configuration()

        # Add nodes with the correct number of outgoing edges
        # First compute neighbors count
        # Each node's outgoing edges = number_of_neighbors
        for g_id in range(N_train):
            for r in range(board_size):
                for c in range(board_size):
                    neighs = hex_neighbors(r, c, board_size=board_size)
                    graphs.add_graph_node(g_id, f"Cell_{r}_{c}", len(neighs))

        graphs.prepare_edge_configuration()

        # Add edges
        for g_id in range(N_train):
            for r in range(board_size):
                for c in range(board_size):
                    node_name = f"Cell_{r}_{c}"
                    neighs = hex_neighbors(r, c, board_size=board_size)
                    for (rr, cc) in neighs:
                        neighbor_name = f"Cell_{rr}_{cc}"
                        graphs.add_graph_node_edge(g_id, node_name, neighbor_name, "Adj")

        # Add node properties (P1, P2, Empty)
        for g_id in range(N_train):
            board = scenario_boards[g_id]
            for idx, val in enumerate(board):
                r = idx // board_size
                c = idx % board_size
                node_name = f"Cell_{r}_{c}"
                if val == 1:
                    symbol = 'P1'
                elif val == -1:
                    symbol = 'P2'
                else:
                    symbol = 'Empty'
                graphs.add_graph_node_property(g_id, node_name, symbol)

        # Encode graphs after adding properties
        graphs.encode()

        # Initialize Tsetlin Machine (similar style to MNIST code)
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            number_of_state_bits=number_of_state_bits
        )

        # Train the Tsetlin Machine
        # We do not have a separate test set, so we measure accuracy on training data
        for epoch in range(epochs):
            tm.fit(graphs, Y, epochs=1, incremental=True)
            predictions = tm.predict(graphs)
            accuracy = (predictions == Y).mean()
            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.2f}")

        # Final accuracy after training
        predictions = tm.predict(graphs)
        accuracy = (predictions == Y).mean()
        print(f"Final Accuracy on {scenario_name}: {accuracy:.2f}")

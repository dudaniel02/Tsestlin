import numpy as np
import pandas as pd
from tqdm import tqdm
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import os

board_size = 11
num_cells = board_size * board_size

# Reduced parameters for faster testing
number_of_clauses = 1
T = 10
s = 1.0
number_of_state_bits = 8
epochs = 1 # fewer epochs to test speed

def load_hex_dataset(csv_path):
    df = pd.read_csv(csv_path)
    board_columns = [f"cell_{i}" for i in range(num_cells)]
    boards = df[board_columns].values
    winners = df['winner'].values
    return boards, winners

def create_partial_states(boards, n_moves):
    partial = boards.copy()
    for i in tqdm(range(partial.shape[0]), desc=f"Creating partial states (removing {n_moves} moves)"):
        row = partial[i]
        non_zero_indices = np.flatnonzero(row)
        if len(non_zero_indices) >= n_moves:
            indices_to_clear = non_zero_indices[-n_moves:]
            row[indices_to_clear] = 0
    return partial

def hex_neighbors(r, c, board_size=11):
    candidates = [
        (r-1, c), (r+1, c),
        (r, c-1), (r, c+1),
        (r-1, c+1), (r+1, c-1)
    ]
    return [(rr, cc) for (rr, cc) in candidates if 0 <= rr < board_size and 0 <= cc < board_size]

def build_graphs(boards, symbols=['P1','P2','Empty']):
    N = boards.shape[0]
    graphs = Graphs(
        N,
        symbols=symbols,
        hypervector_size=64,
        hypervector_bits=2
    )

    # Set number of nodes per graph
    for g_id in tqdm(range(N), desc="Setting graph nodes count"):
        graphs.set_number_of_graph_nodes(g_id, num_cells)
    graphs.prepare_node_configuration()

    # Add nodes
    for g_id in tqdm(range(N), desc="Adding nodes"):
        for r in range(board_size):
            for c in range(board_size):
                neighs = hex_neighbors(r, c, board_size=board_size)
                graphs.add_graph_node(g_id, f"Cell_{r}_{c}", len(neighs))

    graphs.prepare_edge_configuration()

    # Add edges
    for g_id in tqdm(range(N), desc="Adding edges"):
        for r in range(board_size):
            for c in range(board_size):
                node_name = f"Cell_{r}_{c}"
                neighs = hex_neighbors(r, c, board_size=board_size)
                for (rr, cc) in neighs:
                    neighbor_name = f"Cell_{rr}_{cc}"
                    graphs.add_graph_node_edge(g_id, node_name, neighbor_name, "Adj")

    # Add properties
    for g_id in tqdm(range(N), desc="Adding node properties"):
        board = boards[g_id]
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

    # Encode
    tqdm.write("Encoding graphs...")
    graphs.encode()

    return graphs

def train_and_evaluate(boards, winners, scenario_name):
    N = boards.shape[0]
    N = min(N, 10)  # reduce to 100 total for speed
    boards = boards[:N]
    winners = winners[:N]

    split = int(0.8 * N)
    boards_train = boards[:split]
    winners_train = winners[:split]
    boards_test = boards[split:]
    winners_test = winners[split:]

    print(f"\nBuilding graphs for scenario: {scenario_name}")
    graphs_train = build_graphs(boards_train)
    graphs_test = build_graphs(boards_test)

    Y_train = winners_train.astype(np.uint32)
    Y_test = winners_test.astype(np.uint32)

    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        number_of_state_bits=number_of_state_bits
    )
    print("Number of graphs:", graphs_train.number_of_graphs)
    print("Number of nodes in first graph:", graphs_train.number_of_graph_nodes[0])

    print(f"\nTraining on scenario: {scenario_name}")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        print(f"Starting epoch {epoch+1}/{epochs}...")

        # Training for 1 epoch at a time
        print("About to start training...")
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        print("Training step completed.")

        # Evaluate on training set
        train_pred = tm.predict(graphs_train)
        train_acc = (train_pred == Y_train).mean()
        tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc:.4f}")

    # Final evaluation on test set
    test_pred = tm.predict(graphs_test)
    test_acc = (test_pred == Y_test).mean()
    print(f"Test Accuracy on scenario '{scenario_name}': {test_acc:.4f}")

if __name__ == "__main__":
    csv_path = "hex_games.csv"
    boards, winners = load_hex_dataset(csv_path)

    final_boards = boards.copy()
    boards_2_before = create_partial_states(boards, 2)
    boards_5_before = create_partial_states(boards, 5)

    train_and_evaluate(final_boards, winners, "final")
    train_and_evaluate(boards_2_before, winners, "2_moves_before")
    train_and_evaluate(boards_5_before, winners, "5_moves_before")

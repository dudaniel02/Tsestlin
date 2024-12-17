import numpy as np
import random
import os
from graphtsetlinmachine import Graphs, MultiClassGraphTsetlinMachine

# =========================================
# Placeholder function for loading your dataset.
# Adjust this to match your data loading mechanism.
# Should return:
# boards: shape (N, 121) with values in {1, -1, 0}
# winners: shape (N,) with values in {1, -1}
def load_dataset():
    # Replace this dummy code with actual data loading
    # For demonstration, we create a random dataset.
    # In practice, load your 11x11 Hex states and winners.
    N = 500  # Suppose we have 500 samples total (just an example)
    boards = np.random.choice([1, -1, 0], size=(N, 121), p=[0.4, 0.4, 0.2])
    # Random winners
    winners = np.random.choice([1, -1], size=(N,))
    return boards, winners

# =========================================
# Convert winner from {1, -1} to {1,0}
def convert_label(winner):
    return 1 if winner == 1 else 0

# Hex adjacency function
def hex_neighbors(r, c, board_size=11):
    candidates = [
       (r-1, c), (r+1, c),
       (r, c-1), (r, c+1),
       (r-1, c+1), (r+1, c-1)
    ]
    return [(rr, cc) for (rr, cc) in candidates if 0 <= rr < board_size and 0 <= cc < board_size]

# =========================================
# Main code
if __name__ == "__main__":
    # Load your dataset
    boards, winners = load_dataset()
    N_total = boards.shape[0]

    # Train on a smaller portion for testing locally
    N_train = 100  # You can increase this as needed
    boards = boards[:N_train]
    winners = winners[:N_train]

    board_size = 11
    num_nodes = board_size * board_size

    # Symbols for node properties
    symbols = ['P1', 'P2', 'Empty']
    hypervector_size = 128
    hypervector_bits = 2

    # Initialize Graphs
    graphs = Graphs(
        N_train,
        symbols=symbols,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits
    )

    # Set number of nodes for each graph
    for graph_id in range(N_train):
        graphs.set_number_of_graph_nodes(graph_id, num_nodes)
    graphs.prepare_node_configuration()

    # Add nodes for each graph
    for graph_id in range(N_train):
        for r in range(board_size):
            for c in range(board_size):
                node_name = f"Cell_{r}_{c}"
                # We'll set edges later, so initially zero outgoing
                graphs.add_graph_node(graph_id, node_name, 0)
    graphs.prepare_edge_configuration()

    # Add edges
    edge_type = "Adj"
    for graph_id in range(N_train):
        for r in range(board_size):
            for c in range(board_size):
                node_name = f"Cell_{r}_{c}"
                neighs = hex_neighbors(r, c, board_size=board_size)
                for (rr, cc) in neighs:
                    neigh_name = f"Cell_{rr}_{cc}"
                    graphs.add_graph_node_edge(graph_id, node_name, neigh_name, edge_type)

    # Assign properties and labels
    Y = np.empty(N_train, dtype=np.uint32)
    for graph_id in range(N_train):
        board = boards[graph_id]
        winner = winners[graph_id]
        Y[graph_id] = convert_label(winner)
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
            graphs.add_graph_node_property(graph_id, node_name, symbol)

    # Set up a training/test split within our small dataset (e.g., 80/20)
    split = int(N_train * 0.8)
    train_idx = np.arange(split)
    test_idx = np.arange(split, N_train)

    # Extract train/test
    # The GraphTsetlinMachine uses entire graphs as input. We can slice directly.
    # However, slicing 'graphs' directly isn't supported as a typical numpy array.
    # We handle training and testing by referencing indices and passing them to fit/evaluate.
    # Some versions of the API support subsampling via arguments. If not, we might need two Graphs objects.
    # For simplicity, let's just train on all, then evaluate on a subset. In practice, create two Graphs objects.
    
    # For demonstration, let's just train on the whole small set and evaluate on the same.
    # In a real scenario, you'd create a separate Graphs object for the test set or implement indexing.

    # Initialize MultiClass Graph Tsetlin Machine
    # For a binary classification: number_of_classes=2
    tm = MultiClassGraphTsetlinMachine(
        number_of_classes=2,
        T=500,
        s=10.0,
        number_of_clauses=200,
        number_of_state_bits=8,
        max_steps=2,   # message passing rounds
        # If you want CPU-only training, set these:
        # set 'cuda=False' if needed, or ensure no CUDA device is visible.
        # By default, it tries GPU if available.
        cuda=(os.environ.get("CUDA_VISIBLE_DEVICES") is not None)  # heuristic
    )

    # Train
    print("Starting training...")
    tm.fit(graphs, Y, epochs=5, incremental=False)  # A few epochs for demonstration
    print("Training completed.")

    # Evaluate on training set itself (just to show evaluation mechanics)
    predictions = tm.predict(graphs)
    accuracy = (predictions == Y).mean()
    print("Training Accuracy:", accuracy)

    # In practice, you would:
    # 1) Create a separate test set (graphs_test, Y_test)
    # 2) Evaluate: test_predictions = tm.predict(graphs_test)
    # 3) Compute test accuracy.

    # Done.

import numpy as np
import pandas as pd
import os
import kagglehub
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

import kagglehub
import shutil
import os

def download_dataset():
    """
    Download the Hex dataset using KaggleHub.
    Returns:
        str: Path to the downloaded CSV file.
    """
    print("Downloading dataset...")
    # Download dataset to the default location
    path = kagglehub.dataset_download("cholling/game-of-hex")
    
    # The default path where the dataset is downloaded
    csv_source_path = os.path.join(path, "hex_games_1_000_000_size_7.csv")
    target_dir = "data/"
    
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Move the file to the target directory
    csv_target_path = os.path.join(target_dir, "hex_games_1_000_000_size_7.csv")
    shutil.move(csv_source_path, csv_target_path)
    
    print("Dataset downloaded and moved to:", csv_target_path)
    return csv_target_path

def load_hex_dataset(csv_path):
    """
    Load Hex board states and winner outcomes from a CSV file.
    Args:
        csv_path (str): Path to the dataset CSV file.
    Returns:
        boards (np.array): Shape (N, 49) for 7x7 boards flattened.
        winners (np.array): Shape (N,) with values {1, -1}.
    """
    df = pd.read_csv(csv_path)
    board_columns = [col for col in df.columns if col.startswith('cell')]
    boards = df[board_columns].values  # Shape: (N, 49) for 7x7 board
    winners = df['winner'].values
    return boards, winners

def convert_label(winner):
    # Convert from {1, -1} to {1, 0}
    return 1 if winner == 1 else 0

def hex_neighbors(r, c, board_size=7):
    # Adjacency in a hex grid for a 7x7 board
    candidates = [
       (r-1, c), (r+1, c),
       (r, c-1), (r, c+1),
       (r-1, c+1), (r+1, c-1)
    ]
    return [(rr, cc) for (rr, cc) in candidates if 0 <= rr < board_size and 0 <= cc < board_size]

if __name__ == "__main__":
    # Download and load dataset
    csv_path = download_dataset()
    boards, winners = load_hex_dataset(csv_path)

    print("Loaded dataset:")
    print("Boards shape:", boards.shape)   # Expecting (N, 49)
    print("Winners shape:", winners.shape) # Expecting (N,)

    # Use a smaller subset for local training
    N_total = boards.shape[0]
    N_train = min(100, N_total)  # Use 100 or fewer if dataset is smaller
    boards = boards[:N_train]
    winners = winners[:N_train]

    board_size = 7
    num_nodes = board_size * board_size

    # Symbols for node properties
    symbols = ['P1', 'P2', 'Empty']
    hypervector_size = 128
    hypervector_bits = 2

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

    # Add nodes
    for graph_id in range(N_train):
        for r in range(board_size):
            for c in range(board_size):
                node_name = f"Cell_{r}_{c}"
                graphs.add_graph_node(graph_id, node_name, 0)
    graphs.prepare_edge_configuration()

    # Add edges based on hex adjacency
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

    # For demonstration, just train and evaluate on the same small dataset
    tm = MultiClassGraphTsetlinMachine(
        number_of_classes=2,
        T=500,
        s=10.0,
        number_of_clauses=200,
        number_of_state_bits=8,
        max_steps=2,
        cuda=(os.environ.get("CUDA_VISIBLE_DEVICES") is not None)  # use GPU if available
    )

    print("Starting training...")
    tm.fit(graphs, Y, epochs=5, incremental=False)
    print("Training completed.")

    predictions = tm.predict(graphs)
    accuracy = (predictions == Y).mean()
    print("Accuracy on training set (small subset):", accuracy)

import argparse
import numpy as np
import pandas as pd
from time import time
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=200, type=int)
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--s", default=5, type=float)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=512, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=16, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

def position_to_edge_id(pos, board_size):
    return pos[0] * board_size + pos[1]

def create_edges_and_counts(board_size):
    edges = []
    for i in range(board_size):
        for j in range(board_size):
            current_node = (i, j)
            # Add neighbors
            if j + 1 < board_size:  # Right neighbor
                edges.append((current_node, (i, j + 1)))
            if i + 1 < board_size:  # Bottom neighbor
                edges.append((current_node, (i + 1, j)))
            if i + 1 < board_size and j - 1 >= 0:  # Bottom-left neighbor
                edges.append((current_node, (i + 1, j - 1)))
            if i + 1 < board_size and j + 1 < board_size:  # Bottom-right neighbor
                edges.append((current_node, (i + 1, j + 1)))
            if j - 1 >= 0:  # Left neighbor
                edges.append((current_node, (i, j - 1)))
            if i - 1 >= 0 and j - 1 >= 0:  # Top-left neighbor
                edges.append((current_node, (i - 1, j - 1)))
            if i - 1 >= 0 and j + 1 < board_size:  # Top-right neighbor
                edges.append((current_node, (i - 1, j + 1)))

    n_edges_list = [0] * (board_size**2)
    for edge in edges:
        n_edges_list[edge[0][0] * board_size + edge[0][1]] += 1
        n_edges_list[edge[1][0] * board_size + edge[1][1]] += 1
    return edges, n_edges_list

def train_and_test(data_path, state_description):
    # Load data
    data = pd.read_csv(data_path)
    board_size = 13
    subset_size = int(data.shape[0] * 0.8)
    test_size = data.shape[0] - subset_size
    X = data.iloc[:subset_size, 0].values
    X_test = data.iloc[subset_size:subset_size + test_size, 0].values
    y = data.iloc[:subset_size, 1].values
    y_test = data.iloc[subset_size:subset_size + test_size, 1].values

    # Prepare symbols
    symbol_names = ["O", "X", " "]

    # Create edges and counts
    edges, n_edges_list = create_edges_and_counts(board_size)

    print(f"\n--- Training for {state_description} ---")
    # Training graph
    graphs_train = Graphs(
        number_of_graphs=subset_size,
        symbols=symbol_names,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing,
    )

    for graph_id in range(X.shape[0]):
        graphs_train.set_number_of_graph_nodes(graph_id=graph_id, number_of_graph_nodes=board_size**2)
    graphs_train.prepare_node_configuration()

    for graph_id in range(X.shape[0]):
        for k in range(board_size**2):
            graphs_train.add_graph_node(graph_id, k, n_edges_list[k])
    graphs_train.prepare_edge_configuration()

    for graph_id in range(X.shape[0]):
        for k in range(board_size**2):
            sym = X[graph_id][k]
            graphs_train.add_graph_node_property(graph_id, k, sym)
        for edge in edges:
            node_id = position_to_edge_id(edge[0], board_size)
            destination_node_id = position_to_edge_id(edge[1], board_size)
            graphs_train.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
            graphs_train.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)
    graphs_train.encode()

    # Test graph
    graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)

    for graph_id in range(X_test.shape[0]):
        graphs_test.set_number_of_graph_nodes(graph_id=graph_id, number_of_graph_nodes=board_size**2)
    graphs_test.prepare_node_configuration()

    for graph_id in range(X_test.shape[0]):
        for k in range(board_size**2):
            graphs_test.add_graph_node(graph_id, k, n_edges_list[k])
    graphs_test.prepare_edge_configuration()

    for graph_id in range(X_test.shape[0]):
        for k in range(board_size**2):
            sym = X_test[graph_id][k]
            graphs_test.add_graph_node_property(graph_id, k, sym)
        for edge in edges:
            node_id = position_to_edge_id(edge[0], board_size)
            destination_node_id = position_to_edge_id(edge[1], board_size)
            graphs_test.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
            graphs_test.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)
    graphs_test.encode()

    # Train the Tsetlin Machine
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        grid=(16*board_size, 1, 1),
        block=(128, 1, 1)
    )

    start_training = time()
    for i in range(args.epochs):
        tm.fit(graphs_train, y, epochs=1, incremental=True)
        train_accuracy = np.mean(y == tm.predict(graphs_train))
        test_accuracy = np.mean(y_test == tm.predict(graphs_test))
        print(f"Epoch#{i+1} -- Accuracy train: {train_accuracy:.4f} -- Accuracy test: {test_accuracy:.4f}")
    stop_training = time()

    # Final Accuracy Calculation
    final_train_accuracy = np.mean(y == tm.predict(graphs_train))
    final_test_accuracy = np.mean(y_test == tm.predict(graphs_test))
    print("\nFinal Accuracy:")
    print(f"Overall Train Accuracy: {final_train_accuracy:.4f}")
    print(f"Overall Test Accuracy: {final_test_accuracy:.4f}")
    print(f"Training Time: {stop_training - start_training:.2f} seconds")


# 1. Train and test on final states
train_and_test('10_hex_13x13.csv', 'Final State')

# 2. Train and test on states 2 moves before end
train_and_test('10_hex_13x13_2moves.csv', '2 Moves Before End')

# 3. Train and test on states 5 moves before end
train_and_test('10_hex_13x13_5moves.csv', '5 Moves Before End')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import torch

# Define project structure variables
BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / "data" / "data_gnn"
ALL_DIR = DATA_DIR / "all"
SAVE_DIR = ALL_DIR / "geometric_pkl"
PLOT_DIR = BASE_DIR / "results" / "graph_plots"

# Create directory for storing plots
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load temporal graphs
with open(SAVE_DIR / "temporal_graphs.pkl", "rb") as f:
    temporal_graphs = pickle.load(f)

print(f"Loaded {len(temporal_graphs)} temporal graphs")

# City names for labeling
cities = ["amsterdam", "rotterdam", "utrecht"]


# Function to convert PyG graph to NetworkX graph
def pyg_to_nx(pyg_graph, timestep=0):
    G = nx.Graph()

    # Add nodes with features at the specified timestep
    for i, features in enumerate(pyg_graph.x):
        # Take the features at the specified timestep
        node_features = features[timestep].detach().numpy()
        city_name = cities[i]
        # First feature is usually NO2
        no2_value = node_features[0]
        G.add_node(i, label=city_name, no2=float(no2_value))

    # Add edges
    edge_index = pyg_graph.edge_index.t().numpy()
    if pyg_graph.edge_attr is not None:
        edge_attr = pyg_graph.edge_attr.detach().numpy()
        for j, (src, dst) in enumerate(edge_index):
            G.add_edge(src, dst, weight=float(edge_attr[j]))
    else:
        for src, dst in edge_index:
            G.add_edge(src, dst)

    return G


# Plot a few sample graphs
def plot_sample_graphs(num_samples=3):
    # Choose random samples
    if len(temporal_graphs) > num_samples:
        sample_indices = np.random.choice(
            len(temporal_graphs), num_samples, replace=False
        )
    else:
        sample_indices = range(len(temporal_graphs))

    for idx in sample_indices:
        graph = temporal_graphs[idx]

        # Plot for first timestep
        plt.figure(figsize=(10, 8))

        # Create NetworkX graph from PyG graph
        G = pyg_to_nx(graph, timestep=0)

        # Create positions for nodes
        pos = {
            0: (0, 0),  # Amsterdam
            1: (0, -1),  # Rotterdam
            2: (1, -0.5),  # Utrecht
        }

        # Get NO2 values for node colors
        node_colors = [G.nodes[n]["no2"] for n in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=800,
            cmap=plt.cm.YlOrRd,
            vmin=min(node_colors),
            vmax=max(node_colors),
        )

        # Draw edges with weights
        edge_weights = [G.edges[e]["weight"] for e in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)

        # Add city labels
        labels = {i: city for i, city in enumerate(cities)}
        nx.draw_networkx_labels(G, pos, labels, font_size=12)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd,
            norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label("NO2 concentration")

        plt.title(f"Temporal Graph {idx} - Timestep 0")
        plt.axis("off")

        # Save the plot
        plt.savefig(
            PLOT_DIR / f"graph_{idx}_timestep_0.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Plot evolution of NO2 over time for each city
        plt.figure(figsize=(12, 6))

        # Extract NO2 values for all timesteps
        no2_values = {}
        for i, city in enumerate(cities):
            no2_values[city] = [
                graph.x[i, t, 0].item() for t in range(graph.x.shape[1])
            ]

        # Plot NO2 values over time
        for city, values in no2_values.items():
            plt.plot(values, label=city)

        plt.title(f"NO2 Evolution - Graph {idx}")
        plt.xlabel("Timestep")
        plt.ylabel("NO2 concentration")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(
            PLOT_DIR / f"graph_{idx}_no2_evolution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    # Display summary info about the graphs
    sample_graph = temporal_graphs[0]
    print(f"Number of nodes: {sample_graph.x.shape[0]}")
    print(f"Number of timesteps: {sample_graph.x.shape[1]}")
    print(f"Number of features per node: {sample_graph.x.shape[2]}")
    print(f"Number of target timesteps: {sample_graph.y.shape[0]}")

    # Plot some sample graphs
    plot_sample_graphs(num_samples=5)
    print(f"Plots saved to {PLOT_DIR}")

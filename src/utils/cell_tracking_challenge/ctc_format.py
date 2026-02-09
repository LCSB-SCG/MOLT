from collections import defaultdict
import networkx as nx
import tifffile
import numpy as np

def write_graph_to_ctc(graph: nx.Graph, output_filename):
    """
    Writes a networkx.Graph to a Cell Tracking Challenge format file.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        output_filename (str): Output filename for the CTC format.
    """
    # Group nodes by tracked_id to form tracks
    tracks = defaultdict(list)
    for node in graph.nodes(data=True):
        node_id, attrs = node
        tracked_id = attrs['tracked_id']
        timepoint = attrs['timepoint']
        tracks[tracked_id].append(timepoint)

    # For each track, find the min and max timepoint (B and E)
    track_info = []
    for tracked_id, timepoints in tracks.items():
        B = min(timepoints)
        E = max(timepoints)
        track_info.append((tracked_id, B, E))

    # Determine parent tracks (if any)
    parent_info = {tracked_id: 0 for tracked_id, _, _ in track_info}

    # If the graph is directed, use edges to determine parent-child relationships
    if graph.is_directed():
        for u, v in graph.edges():
            u_tracked_id = graph.nodes[u]['tracked_id']
            v_tracked_id = graph.nodes[v]['tracked_id']
            if u_tracked_id != v_tracked_id:
                parent_info[v_tracked_id] = u_tracked_id

    # Write to file
    with open(output_filename, 'w') as f:
        for tracked_id, B, E in track_info:
            P = parent_info[tracked_id]
            f.write(f"{tracked_id} {B} {E} {P}\n")

def write_to_tiff(tiff_data, filename):
    """
    Save a 3D numpy array to a TIFF file.

    Args:
    tiff_data (np.ndarray): 3D numpy array to save as TIFF.
    filename (str): Path to the file to save the TIFF to.
    """
    # If the data is 3D but z=1, squeeze or reshape to 2D if needed
    tiff_data = tiff_data.astype(np.uint16)
    tifffile.imwrite(filename, tiff_data)

# Example usage:
if __name__ == "__main__":
    G = nx.read_graphml("/workspaces/MOLT/results/Fluo-N2DL-HeLa/lineage_graphs/Cells_01_001_graph.graphml")  # Load your graph
    write_graph_to_ctc(G, "/workspaces/MOLT/results/Fluo-N2DL-HeLa/lineage_graphs/Cells_01_001_graph.txt")

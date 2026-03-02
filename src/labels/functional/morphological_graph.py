"""
Morphological Graph Module

This module builds and processes morphological graphs for instance tracking.
The morphological graph represents temporal relationships between instances
across time points, enabling tracking of splits and merges.

Key Concepts:
- Union Nodes: Intermediate nodes representing connected components in the 
  binary union (format: "U_{time_idx}_{union_id}")
- Instance Nodes: Original instances from each time point (format: "{time_idx}_{instance_id}")
- Edges: Connect instances that overlap through union components
- TCUI: Temporally Consistent Unique Instance - unique ID that persists across time,
        getting new IDs when splits or merges occur

The graph is a Directed Acyclic Graph (DAG) when properly constructed, allowing
for efficient traversal to assign unique tracking IDs.
"""

from typing import Dict, Tuple
import networkx as nx


def enforce_no_merge_split(
    graph: nx.Graph,
    enforce_no_merge: bool = False,
    enforce_no_split: bool = False,
):
    """
    Remove edges that would cause merges or splits based on IOU values.

    When enforce_no_merge is True, if a node has multiple incoming edges
    (indicating a merge), only the edge with highest IOU is kept.

    When enforce_no_split is True, if a node has multiple outgoing edges
    (indicating a split), only the edge with highest IOU is kept.

    Args:
        graph: The morphological graph to modify.
        enforce_no_merge: If True, remove edges causing merges.
        enforce_no_split: If True, remove edges causing splits.

    Returns:
        The modified graph.
    """
    if enforce_no_merge:
        # For nodes with multiple incoming edges (merge), keep only highest IOU
        for node in graph.nodes:
            in_edges = list(graph.in_edges(node, data=True))
            if len(in_edges) > 1:
                # Sort by IOU value (descending), keep highest
                in_edges.sort(key=lambda x: x[2].get("iou", 0), reverse=True)
                # Remove all but the first (highest IOU) edge
                for edge in in_edges[1:]:
                    graph.remove_edge(edge[0], edge[1])

    if enforce_no_split:
        # For nodes with multiple outgoing edges (split), keep only highest IOU
        for node in graph.nodes:
            out_edges = list(graph.out_edges(node, data=True))
            if len(out_edges) > 1:
                # Sort by IOU value (descending), keep highest
                out_edges.sort(key=lambda x: x[2].get("iou", 0), reverse=True)
                # Remove all but the first (highest IOU) edge
                for edge in out_edges[1:]:
                    graph.remove_edge(edge[0], edge[1])

    return graph


def get_morphological_graph_global_view_mapping(
    graph: nx.Graph,
    iou: Dict[str, list[Tuple[str, float]]] = None,
    enforce_no_merge: bool = False,
    enforce_no_split: bool = False,
):
    """
    Build a directed morphological graph from the global binary union adjacency graph.

    This function converts the undirected adjacency graph (which contains both
    instance nodes and union nodes) into a directed graph containing only
    instance nodes, connected through time based on the global binary union.

    Algorithm:
    1. Start with an empty directed graph
    2. For each union node in the original graph:
       - Find all instance nodes connected to it
       - Group instances by their time point
       - Connect all instances from time T to all instances from time T+1
       - Store IOU values on edges if available
    3. Add any instance nodes that weren't connected through unions

    The global view considers ALL time points together, so instances that
    touch at ANY point in the series are connected.

    Args:
        graph: Undirected adjacency graph with instance and union nodes.
        iou: Dictionary mapping instance IDs to (match_id, iou_value) pairs.
        enforce_no_merge: If True, prevent merges in the graph.
        enforce_no_split: If True, prevent splits in the graph.

    Returns:
        NetworkX directed graph with only instance nodes, connected through time.
    """
    new_graph = nx.DiGraph()

    # Process each union node to create time-point connections
    for node in graph.nodes:
        if "U_" not in node:
            # Skip instance nodes (we'll add them later)
            continue

        # Parse union node ID to get time index
        # Format: "U_{time_idx}_{union_component_id}"
        _, _, union_id_in_index = node.split("_")
        union_id_in_index = int(union_id_in_index)

        # Get all instance nodes connected to this union
        connected_nodes = [n for n in graph.neighbors(node) if "U_" not in n]

        if len(connected_nodes) == 0:
            continue

        # Group instances by their time point
        timepoints = sorted(list(set([int(n.split("_")[0]) for n in connected_nodes])))
        timepoint_to_nodes = {tp: [] for tp in timepoints}
        for n in connected_nodes:
            timepoint = int(n.split("_")[0])
            timepoint_to_nodes[timepoint].append(n)

        if len(timepoint_to_nodes) == 1:
            # Add the nodes to the graph without edges as they are all from the same time point
            for tp, nodes in timepoint_to_nodes.items():
                for node in nodes:
                    if not new_graph.has_node(node):
                        new_graph.add_node(node)
                        nx.set_node_attributes(
                            new_graph,
                            {node: {"timepoint": tp}},
                        )
        else:
            # Connect instances between consecutive time points
            for i in range(len(timepoints) - 1):
                current_timepoint = timepoints[i]
                next_timepoint = timepoints[i + 1]

                for current_node in timepoint_to_nodes[current_timepoint]:
                    # Add current node if not present
                    if not new_graph.has_node(current_node):
                        new_graph.add_node(current_node)
                        nx.set_node_attributes(
                            new_graph,
                            {current_node: {"timepoint": current_timepoint}},
                        )
                    
                    for next_node in timepoint_to_nodes[next_timepoint]:
                        # Add next node if not present
                        if not new_graph.has_node(next_node):
                            new_graph.add_node(next_node)
                            nx.set_node_attributes(
                                new_graph,
                                {next_node: {"timepoint": next_timepoint}},
                            )
                        
                        # Add edge between consecutive time points
                        new_graph.add_edge(current_node, next_node)
                        
                        # Store IOU value on edge if available
                        if iou and current_node in iou.get("forward_in_time", {}):
                            for o_node, iou_value in iou["forward_in_time"][current_node]:
                                if o_node == next_node:
                                    nx.set_edge_attributes(
                                        new_graph,
                                        {(current_node, next_node): {"iou": iou_value}},
                                    )
                                    break

    # Apply merge/split constraints
    new_graph = enforce_no_merge_split(
        graph=new_graph,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    # This might be wrong
    # # Add any instance nodes that weren't connected through unions
    # # This ensures all instances are represented in the graph
    # for node in graph.nodes:
    #     if "U_" not in node:
    #         if not new_graph.has_node(node):
    #             new_graph.add_node(node)
    #             nx.set_node_attributes(
    #                 new_graph,
    #                 {node: {"timepoint": int(node.split("_")[0])}},
    #             )

    return new_graph


def get_morphological_graph_frame_pair_mapping(
    graph: nx.Graph,
    iou: Dict[str, list[Tuple[str, float]]] = None,
    enforce_no_merge: bool = False,
    enforce_no_split: bool = False,
):
    """
    Build a directed morphological graph using pairwise binary union.

    Unlike the global view which connects all time points together, this
    creates connections only between consecutive time points (frame pairs).

    Algorithm:
    1. Start with an empty directed graph
    2. For each union node:
       - Identify which instances are BEFORE the union time point (incoming)
       - Identify which instances are AT or AFTER the union time point (outgoing)
       - Connect incoming to outgoing instances
    3. Add any instance nodes not connected through unions

    Args:
        graph: Undirected adjacency graph with instance and union nodes.
        iou: Dictionary mapping instance IDs to IOU values.
        enforce_no_merge: If True, prevent merges.
        enforce_no_split: If True, prevent splits.

    Returns:
        NetworkX directed graph with only instance nodes.
    """
    new_graph = nx.DiGraph()

    # Process each union node
    for node in graph.nodes:
        if "U_" not in node:
            continue

        # Parse union node to get time index
        # Format: "U_{pair_idx}_{union_id}"
        _, union_view_idx, union_id_in_index = node.split("_")
        union_view_idx = int(union_view_idx)
        union_id_in_index = int(union_id_in_index)

        # Get all connected instance nodes
        connected_nodes = [n for n in graph.neighbors(node)]

        # Separate into incoming (before union time) and outgoing (at/after union time)
        in_nodes = [
            n for n in connected_nodes if int(n.split("_")[0]) < union_view_idx
        ]
        out_nodes = [
            n for n in connected_nodes if int(n.split("_")[0]) >= union_view_idx
        ]

        # Add all outgoing nodes to graph
        for out_node in out_nodes:
            if not new_graph.has_node(out_node):
                new_graph.add_node(out_node)
                nx.set_node_attributes(
                    new_graph,
                    {out_node: {"timepoint": int(out_node.split("_")[0])}},
                )

        # Connect incoming to outgoing nodes
        for in_node in in_nodes:
            in_node_img_idx = int(in_node.split("_")[0])
            if not new_graph.has_node(in_node):
                new_graph.add_node(in_node)
                nx.set_node_attributes(
                    new_graph,
                    {in_node: {"timepoint": in_node_img_idx}},
                )
            
            for out_node in out_nodes:
                new_graph.add_edge(in_node, out_node)
                
                # Add IOU value if available
                if iou and in_node in iou.get("forward_in_time", {}):
                    for o_node, iou_value in iou["forward_in_time"][in_node]:
                        if o_node == out_node:
                            nx.set_edge_attributes(
                                new_graph,
                                {(in_node, out_node): {"iou": iou_value}},
                            )
                            break

    # Apply merge/split constraints
    new_graph = enforce_no_merge_split(
        graph=new_graph,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    # # Add any instance nodes not connected through unions
    # for node in graph.nodes:
    #     if "U_" not in node:
    #         if not new_graph.has_node(node):
    #             new_graph.add_node(node)
    #             nx.set_node_attributes(
    #                 new_graph,
    #                 {node: {"timepoint": int(node.split("_")[0])}},
    #             )

    return new_graph


def assign_tcui_ids(morphological_graph: nx.Graph):
    """
    Assign Temporally Consistent Unique Instance (TCUI) IDs to graph nodes.

    This traverses the morphological graph and assigns unique tracking IDs
    that persist across time. The algorithm:

    1. Starts at root nodes (nodes with no incoming edges)
    2. Traverses forward through the graph
    3. For linear paths, keeps the same TCUI ID
    4. For splits (one parent -> multiple children), assigns NEW TCUI IDs
       to each child branch
    5. For merges (multiple parents -> one child), assigns a NEW TCUI ID
       to the merged result

    This ensures that:
    - A single cell lineage maintains the same ID as long as it doesn't split
    - When a cell divides (split), each daughter gets a unique new ID
    - When cells merge, the result gets a new unique ID

    Args:
        morphological_graph: NetworkX directed graph with timepoint attributes.

    Returns:
        Tuple of (modified_graph, node_to_tcui_mapping).
        - modified_graph: The input graph with "tracked_id" attribute on each node
        - node_to_tcui_mapping: Dictionary mapping node IDs to TCUI IDs
    """
    from collections import deque

    # Find root nodes (no incoming edges) - these start new lineages
    root_nodes = [
        node
        for node in morphological_graph.nodes
        if morphological_graph.in_degree(node) == 0
    ]

    # Handle edge case: no root nodes (shouldn't happen in valid tracking graphs)
    if not root_nodes:
        root_nodes = list(morphological_graph.nodes)

    current_id = 0
    mapping = {}

    # BFS traversal of the graph
    nodes_to_visit = deque(root_nodes)
    visited = set()

    while len(nodes_to_visit) > 0:
        current_node = nodes_to_visit.popleft()

        if current_node in visited:
            continue
        visited.add(current_node)

        # Get incoming edges to determine if this is root, continuation, split, or merge
        in_edges = morphological_graph.in_edges(current_node, data=True)

        if len(in_edges) == 0:
            # Root node - start a new lineage with new ID
            current_id += 1
            morphological_graph.nodes[current_node]["tracked_id"] = current_id
            mapping[current_node] = current_id

        elif len(in_edges) == 1:
            # Single parent - check if it's a continuation or split
            incoming_node = list(in_edges)[0][0]
            num_out_edges_of_incoming = morphological_graph.out_degree(incoming_node)

            if num_out_edges_of_incoming == 1:
                # Single parent with single child - continuation, keep same ID
                morphological_graph.nodes[current_node]["tracked_id"] = (
                    morphological_graph.nodes[incoming_node]["tracked_id"]
                )
            else:
                # Split detected (one parent, multiple children)
                # Assign new ID to this branch
                current_id += 1
                morphological_graph.nodes[current_node]["tracked_id"] = current_id
            mapping[current_node] = morphological_graph.nodes[current_node][
                "tracked_id"
            ]

        else:
            # Merge detected (multiple parents converge to one child)
            # Assign new ID to the merged result
            current_id += 1
            morphological_graph.nodes[current_node]["tracked_id"] = current_id
            mapping[current_node] = current_id

        # Queue outgoing nodes for processing
        for out_node in morphological_graph.successors(current_node):
            if out_node not in visited:
                nodes_to_visit.append(out_node)

    # # Handle any nodes that weren't visited (isolated or unreachable)
    # for node in morphological_graph.nodes:
    #     if node not in mapping:
    #         current_id += 1
    #         mapping[node] = current_id
    #         morphological_graph.nodes[node]["tracked_id"] = current_id

    return morphological_graph, mapping

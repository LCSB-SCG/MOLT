from typing import Dict, Tuple
import networkx as nx


def enforce_no_merge_split(
    graph: nx.Graph,
    enforce_no_merge: bool = False,
    enforce_no_split: bool = False,
):
    """
    Enforce no merge and no split in the morphological graph.
    This is done by removing edges that would cause a merge or a split.
    """
    if enforce_no_merge:
        # Remove the edges that would cause a merge, only keep the heighest iou edge
        for node in graph.nodes:
            in_edges = list(graph.in_edges(node, data=True))
            if len(in_edges) > 1:
                # Sort edges by iou value
                in_edges.sort(key=lambda x: x[2]["iou"], reverse=True)
                # Keep only the edge with the highest iou
                for edge in in_edges[1:]:
                    graph.remove_edge(edge[0], edge[1])

    if enforce_no_split:
        # Remove the edges that would cause a split, only keep the heighest iou edge
        for node in graph.nodes:
            out_edges = list(graph.out_edges(node, data=True))
            if len(out_edges) > 1:
                # Sort edges by iou value
                out_edges.sort(key=lambda x: x[2]["iou"], reverse=True)
                # Keep only the edge with the highest iou
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
    Return a morphological graph that removes all global view union ids.
    """
    # Replace all union_ids and add skip connections
    new_graph = nx.DiGraph()

    # Remove all union nodes and connect their incoming and outgoing edges
    for node in graph.nodes:
        if "U_" in node:
            # Connect all the connected nodes connected to this union node
            # by sorting them in the order of their timepoint and adding edges between them"
            _, _, union_id_in_index = node.split("_")
            # idx will be the the first timepoint as the registration is done to the first timepoint
            # only the id is important
            union_id_in_index = int(union_id_in_index)

            # get all nodes connected to this union node
            connected_nodes = [n for n in graph.neighbors(node)]

            # sort the connected nodes by their timepoint
            # get a sorted list of all timepoints of the connected nodes
            timepoints = sorted([int(n.split("_")[0]) for n in connected_nodes])
            # create a mapping of timepoint to list of nodes from that timepoint
            timepoint_to_nodes = {tp: [] for tp in timepoints}
            for n in connected_nodes:
                timepoint = int(n.split("_")[0])
                timepoint_to_nodes[timepoint].append(n)

            # connect all the nodes from one timepoint the nodes of the next timepoint
            for i in range(len(timepoints) - 1):
                current_timepoint = timepoints[i]
                next_timepoint = timepoints[i + 1]

                # connect all nodes of the current timepoint to all nodes of the next timepoint
                for current_node in timepoint_to_nodes[current_timepoint]:
                    # add the current node to the new graph if it does not exist
                    if not new_graph.has_node(current_node):
                        new_graph.add_node(current_node)
                        nx.set_node_attributes(
                            new_graph,
                            {current_node: {"timepoint": current_timepoint}},
                        )
                    for next_node in timepoint_to_nodes[next_timepoint]:
                        # add the next node to the new graph if it does not exist
                        if not new_graph.has_node(next_node):
                            new_graph.add_node(next_node)
                            nx.set_node_attributes(
                                new_graph,
                                {next_node: {"timepoint": next_timepoint}},
                            )
                        new_graph.add_edge(current_node, next_node)
                        # set the iou value for the edge
                        for o_node, iou_value in iou["forward_in_time"][current_node]:
                            if o_node == next_node:
                                nx.set_edge_attributes(
                                    new_graph,
                                    {(current_node, next_node): {"iou": iou_value}},
                                )
                                break

    new_graph = enforce_no_merge_split(
        graph=new_graph,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    return new_graph


def get_morphological_graph_frame_pair_mapping(
    graph: nx.Graph,
    iou: Dict[str, list[Tuple[str, float]]] = None,
    enforce_no_merge: bool = False,
    enforce_no_split: bool = False,
):
    # Replace all union_ids and add skip connections
    new_graph = nx.DiGraph()

    # Remove all union nodes and connect their incoming and outgoing edges
    for node in graph.nodes:
        if "U_" in node:
            _, union_view_idx, union_id_in_index = node.split("_")
            union_view_idx = int(union_view_idx)
            union_id_in_index = int(union_id_in_index)

            # get all nodes connected to this union node
            connected_nodes = [n for n in graph.neighbors(node)]

            # determine the incoming ini_id nodes
            in_nodes = [
                n for n in connected_nodes if int(n.split("_")[0]) < union_view_idx
            ]
            out_nodes = [
                n for n in connected_nodes if int(n.split("_")[0]) >= union_view_idx
            ]

            # add all out nodes to the new graph
            for out_node in out_nodes:
                if not new_graph.has_node(out_node):
                    new_graph.add_node(out_node)
                    nx.set_node_attributes(
                        new_graph,
                        {out_node: {"timepoint": int(out_node.split("_")[0])}},
                    )

            # connect the in nodes to out nodes
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
                    for o_node, iou_value in iou["forward_in_time"][in_node]:
                        if o_node == out_node:
                            nx.set_edge_attributes(
                                new_graph,
                                {(in_node, out_node): {"iou": iou_value}},
                            )
                            break

    new_graph = enforce_no_merge_split(
        graph=new_graph,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    return new_graph


def assign_tcui_ids(morphological_graph: nx.Graph):
    """
    Assign a temporally consistent unique instance id (tcui) to each branch in the morphological graph (has to be DAG).
    This is done by traversing the graph and assigning a unique id to each branch.
    When a branch is split, the new branch gets a new id.
    When a branch is merged, the merged branch gets a new id.
    """
    # get all root nodes in the graph
    root_nodes = [
        node
        for node in morphological_graph.nodes
        if morphological_graph.in_degree(node) == 0
    ]
    current_id = 0
    mapping = {}

    nodes_to_visit = list(root_nodes)
    # traverse the graph and assign ids to each node
    while len(nodes_to_visit) > 0:
        current_node = nodes_to_visit.pop(0)

        in_edges = list(morphological_graph.in_edges(current_node, data=True))

        if len(in_edges) == 0:
            # this is a root node, assign a new id
            current_id += 1
            morphological_graph.nodes[current_node]["tracked_id"] = current_id
            mapping[current_node] = current_id

        if len(in_edges) == 1:
            incoming_node = in_edges[0][0]
            num_out_edges_of_incoming = morphological_graph.out_degree(incoming_node)
            # if the parent node has only one outgoing edge, it is a single branch
            # if the parent node has multiple outgoing edges, it is a split
            if num_out_edges_of_incoming == 1:
                morphological_graph.nodes[current_node]["tracked_id"] = (
                    morphological_graph.nodes[incoming_node]["tracked_id"]
                )
            else:
                # this is a split, assign a new id
                current_id += 1
                morphological_graph.nodes[current_node]["tracked_id"] = current_id
            mapping[current_node] = morphological_graph.nodes[current_node][
                "tracked_id"
            ]

        if len(in_edges) > 1:
            # this is a merge, assign a new id
            current_id += 1
            morphological_graph.nodes[current_node]["tracked_id"] = current_id
            mapping[current_node] = current_id

        # add all outgoing nodes to the list of nodes to visit
        for out_node in morphological_graph.successors(current_node):
            if out_node not in nodes_to_visit:
                nodes_to_visit.append(out_node)

    return morphological_graph, mapping

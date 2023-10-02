import itertools
import math
import numpy as np
import numba
import tskit


spec = [
    ("num_edges", numba.int64),
    ("sequence_length", numba.float64),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("edge_insertion_order", numba.int32[:]),
    ("edge_removal_order", numba.int32[:]),
    ("edge_insertion_index", numba.int64),
    ("edge_removal_index", numba.int64),
    ("interval", numba.float64[:]),
    ("in_range", numba.int64[:]),
    ("out_range", numba.int64[:]),
]


@numba.experimental.jitclass(spec)
class TreePosition:
    def __init__(
        self,
        num_edges,
        sequence_length,
        edges_left,
        edges_right,
        edge_insertion_order,
        edge_removal_order,
    ):
        self.num_edges = num_edges
        self.sequence_length = sequence_length
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.edge_insertion_index = 0
        self.edge_removal_index = 0
        self.interval = np.zeros(2)
        self.in_range = np.zeros(2, dtype=np.int64)
        self.out_range = np.zeros(2, dtype=np.int64)

    def next(self):
        left = self.interval[1]
        j = self.in_range[1]
        k = self.out_range[1]
        self.in_range[0] = j
        self.out_range[0] = k
        M = self.num_edges
        edges_left = self.edges_left
        edges_right = self.edges_right
        out_order = self.edge_removal_order
        in_order = self.edge_insertion_order

        while k < M and edges_right[out_order[k]] == left:
            k += 1
        while j < M and edges_left[in_order[j]] == left:
            j += 1
        self.out_range[1] = k
        self.in_range[1] = j

        right = self.sequence_length
        if j < M:
            right = min(right, edges_left[in_order[j]])
        if k < M:
            right = min(right, edges_right[out_order[k]])
        self.interval[:] = [left, right]
        return j < M or left < self.sequence_length


# Helper function to make it easier to communicate with the numba class
def alloc_tree_position(ts):
    return TreePosition(
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
    )


@numba.njit(cache=True)
def _tree_traversal(tree):
    ret = 0
    return ret


@numba.njit(cache=True)
def _track_branchtypes_site(ts, bt_map):
    pass


@numba.njit(cache=True)
def _track_branchtypes_branch(
    tree_pos, 
    edges_parent, 
    edges_child, 
    edges_left, 
    edges_right,
    nodes_time, 
    nodes_bt_array,
    dims,
):
    ret = np.zeros(dims, dtype=np.float64)
    delta_ret = np.zeros(dims[1], dtype=np.int32)
    # initialize node_bt_array using samples
    tree_idx = 0

    while tree_pos.next():
        span = tree_pos.interval[1] - tree_pos.interval[0]
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            # parse edges out
            e = tree_pos.edge_removal_order[j]
            p = edges_parent[e]
            c = edges_child[e]
            depth = nodes_time[p] - nodes_time[e]
            # determine branch type of e
            bte = nodes_bt_array[c]
            nodes_bt_array[p] -= bte
            assert bte != 0
            delta_ret[bte] -= span * depth

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            # parse edges in
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            c = edges_child[e]
            depth = nodes_time[p] - nodes_time[e]
            bte = nodes_bt_array[c]
            nodes_bt_array[p] |= bte
            assert bte != 0
            delta_ret[bte] += span * depth
        
        ret[tree_idx] += delta_ret
        tree_idx += 1

    return ret


def track_branchtypes(ts, bt_map, mode='site'):
    # bt_array should map each sample to a label.
    assert len(bt_map)==ts.num_samples, \
        "The number of entries in bt_map should equal the number \
        of samples in the tree sequence."
    # each entry should correspond to an integer that only has a single 1
    # in its binary representation.
    for s in bt_map:
        assert s == s & (-s), "Map incorrect"
    num_branchtypes = sum(2**i for i in range(ts.num_samples, 0, -1))
    node_bt_array = np.zeros(ts.num_nodes+1, dtype=np.uint32)
    for sample, bt in zip(ts.samples(), bt_map):
        node_bt_array[sample] = bt
    dims = (ts.num_trees, num_branchtypes)

    tree_pos = alloc_tree_position(ts)

    if mode=='sites':
        return _track_branchtypes_site(
            tree_pos,
            ts.edges_parent,
            ts.edges_child,
            ts.edges_left,
            ts.edges_right,
            ts.nodes_time,
            node_bt_array,
            dims
        )
    elif mode=='branch':
        return _track_branchtypes_branch(
            tree_pos,
            ts.edges_parent,
            ts.edges_child,
            ts.edges_left,
            ts.edges_right,
            ts.nodes_time,
            node_bt_array,
            dims
        )
    else:
        raise ValueError(f'Either specify site or branch as mode.')
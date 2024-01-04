import collections
import itertools
import numpy as np

import agemo.events as eventslib

# auxilliary functions


def flatten(input_list):
    """
    Flattens iterable from depth n to depth n-1
    """
    return itertools.chain.from_iterable(input_list)


# representing samples
def sample_to_str(sample_list):
    return "/".join("_".join(lineage for lineage in pop) for pop in sample_list)


def paths_from_visited_node(graph, node, equation_dict, path):
    stack = [
        (path, node),
    ]
    while stack:
        path, parent = stack.pop()
        children = graph.get(parent, None)
        if children is not None:
            for child in children:
                stack.append(
                    (
                        path[:]
                        + [
                            equation_dict[(parent, child)],
                        ],
                        child,
                    )
                )
        else:
            yield path


class GfMatrixObject:
    """

    :param branch_type_counter: Object describing all branchtypes for which to return
       the Laplace transformed joint distribution of coalescence times.
    :type branch_type_counter: class `agemo.BranchTypeCounter`
    :param events: List of events defining the structured coalescent model, defaults to None.
    :type events: list(class `agemo.Event`), optional

    """

    def __init__(
        self,
        branch_type_counter,
        events=None,
    ):

        self.sample_list = branch_type_counter.sample_configuration
        self.branchtype_dict = branch_type_counter.labels_dict
        self.num_branchtypes = len(branch_type_counter)
        if events is None:
            events = []
        self.optional_events = events

        all_event_idxs, self.discrete_events = [], []
        for event in events:
            if event.discrete:
                self.discrete_events.append(event.idx)
            all_event_idxs.append(event.idx)
        num_events = len(events)
        num_coalescence_events = len(self.sample_list)
        self.num_variables = num_coalescence_events + num_events
        assert sorted(all_event_idxs) == list(
            range(num_coalescence_events, self.num_variables)
        ), f"all event idxs should be unique and range between {num_coalescence_events} and {self.num_variables}."

        self.discrete_events = [event.idx for event in events if event.discrete]
        self.coalescence_events = eventslib.CoalescenceEventsSuite(
            len(self.sample_list)
        )

    def make_gf(self):
        """
        Build the generating function recursively given the specified structured coalescent
        model and a branchtype mapping. Returns a tuple consisting of all paths and all
        equations in matrix form. The paths are a list of list, where each list describes
        a single path. The indices for each path point to the index of the equation in the
        equation array.

        :return gf: paths, equations
        :rtype gf: (list(list(int)), class `np.ndarray`)

        """
        stack = [
            (list(), self.sample_list),
        ]
        paths = list()
        eq_list = list()
        eq_idx = 0
        # keeping track of things
        graph_dict = collections.defaultdict(list)
        equation_dict = dict()  # key=(parent, child), value=eq_idx
        nodes_visited = set()  # list of all nodes visisted

        while stack:
            path_so_far, state = stack.pop()
            parent_node = sample_to_str(state)
            if sum(len(pop) for pop in state) == 1:
                paths.append(path_so_far)
            else:
                if parent_node in nodes_visited:
                    # depth first search through graph
                    for add_on_path in paths_from_visited_node(
                        graph_dict, parent_node, equation_dict, path_so_far
                    ):
                        paths.append(add_on_path)
                else:
                    nodes_visited.add(parent_node)
                    multiplier_array, new_state_list = self.gf_single_step(state)
                    eq_list.append(multiplier_array)
                    for new_state in new_state_list:
                        child_node = sample_to_str(new_state)
                        path = path_so_far[:]
                        path.append(eq_idx)
                        graph_dict[parent_node].append(child_node)
                        equation_dict[(parent_node, child_node)] = eq_idx
                        stack.append((path, new_state))
                        eq_idx += 1
        return (paths, np.concatenate(eq_list, axis=0))


    def equations_graph(self):
        """
        Returns all information needed to evaluate the generating function.
        `adjacency_list`: contains a tuple with the children of each parent i at index i.
        `node_to_equations_map`: tuple pointing to equation in equation matrix represented by
        each node. Note that only nodes that require an inverse Laplace transform with
        respect to a discrete event might be associated with a tuple containing more than one
        integer.
        `to_invert_array` is a boolean array indicating whether the equations associated with
        the node at each index require an inverse Lapalce transform with respect to a discrete
        event.
        `equation_matrix`: array containing the coefficients for the numerator and denominator
        of the uninverted Laplace transform of the joint coalescence time distribution. This
        is the same matrix as would have been obtained by running `make_gf()`.

        :return computational_graph: (`adjacency_list`, `node_to_equations_map`, `to_invert_array`, `equation_matrix`)
        :rtype computational_graph: (list(tuple(int)), , class `np.ndarray(bool)`, class `np.ndarray(int)`)

        """
        num_discrete_events = len(self.discrete_events)
        if num_discrete_events:
            if num_discrete_events > 1:
                raise NotImplementedError
            else:
                delta_idx = self.discrete_events[0]
        else:
            delta_idx = None

        stack = [
            (list(), self.sample_list, 0),
        ]
        eq_list = list()
        eq_idx = 0
        node_idx, inverted_node_idx = 1, -1
        # keeping track of things
        eq_graph_dict = collections.defaultdict(
            set
        )  # key = parent_eq_node, value = [children eq_nodes]
        equation_dict = dict()  # key=eq_node, value=[eq_idx1, eq_idx2,...]
        state_eq_dict = dict()  # key=(parent, child), value=eq_idx
        state_node_dict = dict()
        visited = set()  # list of all states visisted

        while stack:
            path_so_far, state, parent_idx = stack.pop()
            parent = sample_to_str(state)
            if sum(len(pop) for pop in state) == 1:
                # add last equation to equation_chain
                # make node of equation chain within eq_graph_dict
                if len(path_so_far) > 0:
                    # equation_dict[inverted_node_idx] = (tuple(path_so_far), True)
                    equation_dict[inverted_node_idx] = tuple(path_so_far)
                    eq_graph_dict[parent_idx].add(inverted_node_idx)
                    inverted_node_idx -= 1
            else:
                parent_visited = parent in visited
                multiplier_array, new_state_list = self.gf_single_step(state)
                if not parent_visited:
                    visited.add(parent)
                    eq_list.append(multiplier_array)
                for new_eq, new_state in zip(multiplier_array, new_state_list):
                    child = sample_to_str(new_state)
                    if delta_idx is not None and new_eq[1, delta_idx] > 0:
                        # equation to be inverted
                        path = path_so_far[:]
                        new_eq_idx = (
                            eq_idx
                            if not parent_visited
                            else state_eq_dict[(parent, child)]
                        )
                        path.append(new_eq_idx)

                        if new_eq[0, delta_idx] > 0:
                            # end of equation chain, make node
                            # equation_dict[inverted_node_idx] = (tuple(path), True)
                            equation_dict[inverted_node_idx] = tuple(path)
                            eq_graph_dict[parent_idx].add(inverted_node_idx)
                            stack.append(([], new_state, inverted_node_idx))
                            inverted_node_idx -= 1
                        else:
                            # equation chain needs to be continued
                            stack.append((path, new_state, parent_idx))
                    else:
                        # single equation will become new node, no delta
                        assert len(path_so_far) == 0
                        if parent_visited:
                            # check if already connected to graph
                            # have to link parent_child pair to node_idx
                            new_node_idx = state_node_dict[(parent, child)]
                            eq_graph_dict[parent_idx].add(new_node_idx)
                        else:
                            stack.append(([], new_state, node_idx))
                            # equation_dict[node_idx] = ((eq_idx,), False)
                            equation_dict[node_idx] = (eq_idx,)
                            eq_graph_dict[parent_idx].add(node_idx)
                            state_node_dict[(parent, child)] = node_idx
                            node_idx += 1

                    if not parent_visited:
                        state_eq_dict[(parent, child)] = eq_idx
                        eq_idx += 1

        # eq_graph_array, eq_array, to_invert_array, eq_matrix
        # array representation of graph, linking nodes to eqs , boolean_to_invert, matrix_with_coefficients
        return (
            *remap_eq_arrays(eq_graph_dict, equation_dict, node_idx, inverted_node_idx),
            np.concatenate(eq_list, axis=0),
        )

    def gf_single_step(self, state_list):
        current_branches = list(flatten(state_list))
        numLineages = len(current_branches)
        if numLineages == 1:
            ValueError(
                "gf_single_step fed with single lineage, should have been caught."
            )

        # collecting the idxs of branches in state_list
        dummy_sum = [self.branchtype_dict[b] for b in current_branches]
        dummy_array = np.zeros((2, self.num_branchtypes), dtype=np.uint8)
        dummy_unique, dummy_counts = np.unique(dummy_sum, return_counts=True)
        dummy_array[1, dummy_unique] = dummy_counts

        outcomes = self.rates_and_events(state_list)
        multiplier_array = np.zeros(
            (len(outcomes), 2, self.num_variables), dtype=np.uint8
        )
        new_state_list = list()

        for event_idx, (rate_idx, count, new_state) in enumerate(outcomes):
            multiplier_array[event_idx, 0, rate_idx] = count
            new_state_list.append(new_state)

        multiplier_array[:, 1] = np.sum(multiplier_array[:, 0], axis=0)
        dummy_array = np.tile(dummy_array, (multiplier_array.shape[0], 1, 1))
        multiplier_array = np.concatenate((multiplier_array, dummy_array), axis=2)
        return (multiplier_array, new_state_list)

    def rates_and_events(self, state_list):
        all_events = []
        all_events += self.coalescence_events._single_step(state_list)
        for event in self.optional_events:
            all_events += event._single_step(state_list)
        return all_events


def remap_eq_arrays(eq_graph_dict, equation_dict, node_idx, inverted_node_idx):
    # if no non_inverted_eq, node_idx==1
    # if no inverted_eq, node_idx==-1
    # remap eq_graph_dict
    max_node_idx = node_idx - 1
    total_num_nodes = max_node_idx - inverted_node_idx - 1
    eq_graph_dict = {
        map_to_pos_values(k, max_node_idx): [
            map_to_pos_values(v, max_node_idx) for v in vs
        ]
        for k, vs in eq_graph_dict.items()
    }
    # remap equation_dict
    equation_dict = {
        map_to_pos_values(k, max_node_idx): vs for k, vs in equation_dict.items()
    }
    to_invert_array = np.ones(total_num_nodes, dtype=bool)
    to_invert_array[:max_node_idx] = 0
    eq_graph_array = tuple(
        [
            np.array(eq_graph_dict[i], dtype=np.int64)
            if i in eq_graph_dict
            else np.array([], dtype=np.int64)
            for i in range(total_num_nodes + 1)
        ]
    )
    eq_array = [
        tuple(equation_dict[i]) if i in equation_dict else tuple()
        for i in range(1, total_num_nodes + 1)
    ]

    return (eq_graph_array, eq_array, to_invert_array)


def map_to_pos_values(v, max_node_idx):
    if v < 0:
        return max_node_idx - v
    else:
        return v


def equations_from_matrix(multiplier_array, variables_array):
    temp = multiplier_array.dot(variables_array)
    return temp[:, 0] / temp[:, 1]

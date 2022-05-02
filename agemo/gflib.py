import collections
import itertools
import numpy as np

import agemo.events as eventslib

# auxilliary functions


def powerset(iterable):
    """
    returns generator containing all possible subsets of iterable
    """
    s = list(iterable)
    return (
        "".join(sorted(subelement))
        for subelement in (
            itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(len(s) + 1)
            )
        )
    )


def flatten(input_list):
    """
    Flattens iterable from depth n to depth n-1
    """
    return itertools.chain.from_iterable(input_list)


def coalesce_lineages(input_tuple, to_join):
    """
    Joins lineages and returns resulting tuple of lineages for a single pop
    """
    result = list(input_tuple)
    for lineage in to_join:
        result.remove(lineage)
    result.append("".join(sorted(flatten(to_join))))
    result.sort()
    return tuple(result)


def coalesce_single_pop(pop_state):
    """
    For single population generate all possible coalescence events
    param: iterable pop_state: containing all lineages (str) present within pop
    param: float, object coal_rate: rate at which coalescence happens in that pop
    """
    coal_event_pairs = list(itertools.combinations(pop_state, 2))
    coal_counts = collections.Counter(coal_event_pairs)
    for lineages, count in coal_counts.items():
        result = coalesce_lineages(pop_state, lineages)
        yield (count, result)


def sort_mutation_types(branchtypes):
    if isinstance(branchtypes, dict):
        return sorted(set(branchtypes.values()), key=lambda x: str(x))
    elif isinstance(branchtypes, list) or isinstance(branchtypes, tuple):
        return sorted(set(branchtypes), key=lambda x: str(x))
    else:
        raise ValueError(f"sort_mutation_types not implemented for {type(branchtypes)}")


def sum_all_but_idx(ar, idx):
    """
    Summing all values across last axis of array except for values at idx
    """
    return np.sum(ar[:, :idx], axis=-1) + np.sum(ar[:, idx + 1 :], axis=-1)


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


class GfObject:
    def __init__(
        self,
        sample_list,
        coalescence_rates,
        branchtype_dict,
        migration_direction=None,
        migration_rate=None,
        exodus_direction=None,
        exodus_rate=None,
    ):
        assert len(sample_list) == len(coalescence_rates)
        if sum(1 for pop in sample_list if len(pop) > 0) > 1:
            assert (
                migration_direction or exodus_direction
            ), "lineages from different populations cannot coalesce without migration or exodus event."
        self.sample_list = tuple(tuple(sorted(pop)) for pop in sample_list)
        self.branchtype_dict = branchtype_dict

        self.coalescence_rates = coalescence_rates
        self.migration_direction = migration_direction
        if migration_direction and not migration_rate:
            raise ValueError("Migration direction provided but no migration rate.")
        else:
            self.migration_rate = migration_rate
        self.exodus_direction = exodus_direction
        if exodus_direction and not exodus_rate:
            raise ValueError("Migration direction provided but no migration rate.")
        else:
            self.exodus_rate = exodus_rate

    def coalescence_events(self, state_list):
        """
        Returning all possible new population configurations due to coalescence,
        and their respective rates
        :param list state_list: list of population tuples containing lineages (str)
        """
        result = []
        for idx, (pop, rate) in enumerate(zip(state_list, self.coalescence_rates)):
            for count, coal_event in coalesce_single_pop(pop):
                modified_state_list = list(state_list)
                modified_state_list[idx] = coal_event
                result.append((count * rate, tuple(modified_state_list)))
        return result

    def migration_events(self, state_list):
        """
        Returning all possible new population configurations due to migration events,
        and their respective rates, BACKWARDS IN TIME
        :param list state_list: list of population tuples containing lineages (str)
        """
        result = []
        if self.migration_direction:
            for source, destination in self.migration_direction:
                lineage_count = collections.Counter(state_list[source])
                for lineage, count in lineage_count.items():
                    temp = list(state_list)
                    idx = temp[source].index(lineage)
                    temp[source] = tuple(temp[source][:idx] + temp[source][idx + 1 :])
                    temp[destination] = tuple(
                        sorted(
                            list(temp[destination])
                            + [
                                lineage,
                            ]
                        )
                    )
                    result.append((count * self.migration_rate, tuple(temp)))
        return result

    def exodus_events(self, state_list):
        """
        Returning all possible new population configurations due to exodus events,
        and their respective rates: BACKWARDS IN TIME
        :param list state_list: list of population tuples containing lineages (str)
        """
        result = []
        if self.exodus_direction:
            for *source, destination in self.exodus_direction:
                temp = list(state_list)
                sources_joined = tuple(
                    itertools.chain.from_iterable([state_list[idx] for idx in source])
                )
                if len(sources_joined) > 0:
                    temp[destination] = tuple(
                        sorted(state_list[destination] + sources_joined)
                    )
                    for idx in source:
                        temp[idx] = ()
                    result.append((self.exodus_rate, tuple(temp)))
        return result

    def rates_and_events(self, state_list):
        """
        Returning all possible events, and their respective rates
        :param list state_list: list of population tuples
        containing lineages (str)
        """
        c = self.coalescence_events(state_list)
        m = self.migration_events(state_list)
        e = self.exodus_events(state_list)
        return c + m + e

    def gf_single_step(self, gf_old, state_list):
        """
        Yields single (tail) recursion step for the generating function
        :param object gf_old: result from previous recursion step
        :param list state_list: list of population tuples containing
        lineages (str)
        """
        current_branches = list(flatten(state_list))
        numLineages = len(current_branches)
        if numLineages == 1:
            ValueError(
                "gf_single_step fed with single lineage, should have been caught."
            )
        else:
            outcomes = self.rates_and_events(state_list)
            total_rate = sum([rate for rate, state in outcomes])
            dummy_sum = sum(self.branchtype_dict[b] for b in current_branches)
            return [
                (gf_old * rate * 1 / (total_rate + dummy_sum), new_state_list)
                for rate, new_state_list in outcomes
            ]

    def make_gf(self):

        stack = [(1, self.sample_list)]
        # result = []
        while stack:
            gf_n, state_list = stack.pop()
            if sum(len(pop) for pop in state_list) == 1:
                yield gf_n
            else:
                for gf_nplus1 in self.gf_single_step(gf_n, state_list):
                    stack.append(gf_nplus1)


class GfObjectChainRule(GfObject):
    def make_gf(self):
        stack = [
            (list(), self.sample_list),
        ]
        paths = list()
        eq_list = list()

        # keeping track of things
        graph_dict = collections.defaultdict(list)
        equation_dict = dict()  # key=(parent, child), value=eq_idx
        nodes_visited = list()  # list of all nodes visisted

        while stack:
            path_so_far, state_list = stack.pop()
            parent_node = sample_to_str(state_list)
            if sum(len(pop) for pop in state_list) == 1:
                paths.append(path_so_far)
            else:
                if parent_node in nodes_visited:
                    # depth first search through graph
                    for add_on_path in paths_from_visited_node(
                        graph_dict, parent_node, equation_dict, path_so_far
                    ):
                        paths.append(add_on_path)
                else:
                    nodes_visited.append(parent_node)
                    for eq, new_state_list in self.gf_single_step(state_list):
                        child_node = sample_to_str(new_state_list)
                        eq_idx = len(eq_list)
                        eq_list.append(eq)
                        path = path_so_far[:]
                        path.append(eq_idx)
                        graph_dict[parent_node].append(child_node)
                        equation_dict[(parent_node, child_node)] = eq_idx
                        stack.append((path, new_state_list))

        return (paths, np.array(eq_list))

    def gf_single_step(self, state_list):
        current_branches = list(flatten(state_list))
        numLineages = len(current_branches)
        if numLineages == 1:
            ValueError(
                "gf_single_step fed with single lineage, should have been caught."
            )
        else:
            outcomes = self.rates_and_events(state_list)
            total_rate = sum([rate for rate, state in outcomes])
            dummy_sum = sum(self.branchtype_dict[b] for b in current_branches)
            return [
                (rate * 1 / (total_rate + dummy_sum), new_state_list)
                for rate, new_state_list in outcomes
            ]


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

    def make_graph(self):
        stack = [
            (0, self.sample_list),
        ]
        eq_list = list()
        eq_idx = 0
        node_idx = 1
        graph_dict = collections.defaultdict(list)
        equation_dict = dict()  # key=(parent, child), value=eq_idx
        nodes_visited = set()  # set of all nodes visisted
        str_to_numeric_node_dict = {sample_to_str(self.sample_list): 0}

        while stack:
            parent_node_numeric, state = stack.pop()
            parent_node = sample_to_str(state)
            if sum(len(pop) for pop in state) > 1:
                if parent_node not in nodes_visited:
                    nodes_visited.add(parent_node)
                    multiplier_array, new_state_list = self.gf_single_step(state)
                    eq_list.append(multiplier_array)
                    for eq, new_state in zip(multiplier_array, new_state_list):
                        child_node = sample_to_str(new_state)
                        if child_node in str_to_numeric_node_dict:
                            child_node_numeric = str_to_numeric_node_dict[child_node]
                        else:
                            child_node_numeric = node_idx
                            str_to_numeric_node_dict[child_node] = child_node_numeric
                            node_idx += 1
                        graph_dict[parent_node_numeric].append(child_node_numeric)
                        equation_dict[
                            (parent_node_numeric, child_node_numeric)
                        ] = eq_idx
                        stack.append((child_node_numeric, new_state))
                        eq_idx += 1

        graph_array = [
            tuple(graph_dict[i]) if i in graph_dict else tuple()
            for i in range(node_idx)
        ]
        adjacency_matrix = eq_dict_to_adjacency_matrix(equation_dict, node_idx, eq_idx)

        return (graph_array, adjacency_matrix, np.concatenate(eq_list, axis=0))

    def collapse_graph(self, graph_array, adjacency_matrix, eq_matrix):
        num_discrete_events = len(self.discrete_events)

        if num_discrete_events == 0:
            collapsed_graph_array = graph_array
            eq_array = tuple([(i,) for i in range(eq_matrix.shape[0])])
            to_invert_array = np.zeros(len(eq_array), dtype=bool)
        else:
            if num_discrete_events > 1:
                raise NotImplementedError
            delta_idx = self.discrete_events[0]
            root = 0
            visited = {root: 0}  # old_index, new_index
            stack = [(root, root, list())]
            new_node_idx = 1
            collapsed_graph_dict = collections.defaultdict(list)
            equations_dict = dict()

            while stack:
                current_node, parent_new_idx, path_so_far = stack.pop()
                children = graph_array[current_node]
                if len(children) == 0 and len(path_so_far) > 0:
                    child_new_idx = new_node_idx
                    new_node_idx += 1
                    collapsed_graph_dict[parent_new_idx].append(child_new_idx)
                    equations_dict[parent_new_idx, child_new_idx] = (
                        tuple(path_so_far),
                        True,
                    )
                else:
                    for child in children:
                        vertex_eq = adjacency_matrix[current_node, child]
                        path = path_so_far[:]
                        path.append(vertex_eq)
                        to_invert = eq_matrix[vertex_eq, 1, delta_idx] == 1
                        if eq_matrix[vertex_eq, 0, delta_idx] == 0 and to_invert:
                            stack.append((child, parent_new_idx, path))
                        elif eq_matrix[vertex_eq, 0, delta_idx] == 1:
                            child_new_idx = new_node_idx
                            new_node_idx += 1
                            collapsed_graph_dict[parent_new_idx].append(child_new_idx)
                            stack.append((child, child_new_idx, []))
                            equations_dict[parent_new_idx, child_new_idx] = (
                                tuple(path),
                                to_invert,
                            )
                        else:
                            if child in visited:
                                child_new_idx = visited[child]
                            else:
                                child_new_idx = new_node_idx
                                visited[child] = child_new_idx
                                new_node_idx += 1
                                stack.append((child, child_new_idx, []))
                            collapsed_graph_dict[parent_new_idx].append(child_new_idx)
                            equations_dict[parent_new_idx, child_new_idx] = (
                                tuple(path),
                                to_invert,
                            )

            collapsed_graph_array = [
                tuple(collapsed_graph_dict[i]) if i in collapsed_graph_dict else tuple()
                for i in range(new_node_idx)
            ]
            (
                adjacency_matrix,
                eq_array,
                to_invert_array,
            ) = eq_dict_to_adjacency_matrix_collapsed_graph(
                equations_dict, new_node_idx
            )
        return (
            collapsed_graph_array,
            adjacency_matrix,
            eq_array,
            to_invert_array,
        )

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


def eq_dict_to_adjacency_matrix(equation_dict, max_idx, max_value):
    dtype = np.min_scalar_type(max_value + 1)
    fill_value = np.iinfo(dtype).max
    adjacency_matrix = np.full((max_idx, max_idx), fill_value=fill_value, dtype=dtype)
    for (p, c), eq_idx in equation_dict.items():
        adjacency_matrix[p, c] = eq_idx
    return adjacency_matrix


def eq_dict_to_adjacency_matrix_collapsed_graph(equation_dict, max_idx):
    sorted_eq_dict = sorted(equation_dict.items(), key=lambda x: x[1][1])
    eq_array = tuple(t[1][0] for t in sorted_eq_dict)
    max_value = max(flatten(eq_array))
    to_invert = np.array([t[1][1] for t in sorted_eq_dict])
    dtype = np.min_scalar_type(max_value + 1)
    fill_value = np.iinfo(dtype).max
    adjacency_matrix = np.full((max_idx, max_idx), fill_value=fill_value, dtype=dtype)
    for idx, ((p, c), _) in enumerate(sorted_eq_dict):
        adjacency_matrix[p, c] = idx
    return (adjacency_matrix, eq_array, to_invert)


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


def split_paths_laplace(paths, multiplier_array, delta_idx):
    # split paths into no_delta, with_delta
    if delta_idx is None:
        filtered_paths = [
            [np.array(path, dtype=int), np.array([], dtype=int)] for path in paths
        ]
    else:
        filtered_paths = []
        for path in paths:
            condition = multiplier_array[path, 1, delta_idx] == 1
            filtered_paths.append(split(path, condition))
    return filtered_paths


def split(arr_in, condition):
    arr = np.array(arr_in, dtype=int)
    return [arr[~condition], arr[condition]]  # return (constants, to_invert)

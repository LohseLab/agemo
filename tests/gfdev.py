import collections
import itertools

import mpmath
import numpy as np
import sympy

import agemo.gflib as gflib
import agemo.inverse as gfinverse
import agemo.legacy.inverse as linverse


@mpmath.workdps(100)
def get_parameter_dict(coalescence_rates, global_info, sim_config, gf_vars):
    parameter_dict = {}
    reference_pop = global_info["reference_pop"]
    if gf_vars.get("migration_rate"):
        migration_string = (
            "me_A_B" if gf_vars["migration_direction"] == [(1, 2)] else "me_B_A"
        )
        parameter_dict[gf_vars["migration_rate"]] = mpmath.mpf(
            2 * sim_config[migration_string] * sim_config[f"Ne_{reference_pop}"]
        )
    if gf_vars.get("exodus_rate"):
        parameter_dict[sympy.symbols("T", real=True, positive=True)] = mpmath.mpf(
            sim_config["T"] / (2 * sim_config[f"Ne_{reference_pop}"])
        )
    for c, Ne in zip(coalescence_rates, ("Ne_A_B", "Ne_A", "Ne_B")):
        if Ne in sim_config:
            parameter_dict[c] = mpmath.mpf(
                sim_config[f"Ne_{reference_pop}"] / sim_config[Ne]
            )
        else:
            parameter_dict[c] = 0.0
    return parameter_dict


@mpmath.workdps(100)
def get_theta(global_info, sim_config, **kwargs):
    reference_pop = global_info["reference_pop"]
    Ne_ref = sim_config[f"Ne_{reference_pop}"]
    mu = global_info["mu"]
    block_length = global_info["blocklength"]
    return 2 * mpmath.mpf(Ne_ref * mu) * block_length


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


def equations_from_matrix(multiplier_array, variables_array):
    temp = multiplier_array.dot(variables_array)
    return temp[:, 0] / temp[:, 1]


def equations_from_matrix_with_inverse(
    multiplier_array, paths, var_array, time, delta_idx
):
    split_paths = split_paths_laplace(paths, multiplier_array, delta_idx)
    delta_in_nom_all = multiplier_array[:, 0, delta_idx] == 1
    results = np.zeros(len(split_paths), dtype=object)
    subset_no_delta = np.arange(multiplier_array.shape[-1]) != delta_idx
    multiplier_array_no_delta = multiplier_array[:, :, subset_no_delta]
    for idx, (no_delta, with_delta) in enumerate(split_paths):
        delta_in_nom_list = delta_in_nom_all[with_delta]
        inverse = gfinverse.inverse_laplace_single_event(
            multiplier_array_no_delta[with_delta],
            var_array,
            time,
            delta_in_nom_list,
        )
        if isinstance(inverse, np.ndarray):
            inverse = np.sum(inverse)
        no_inverse = np.prod(
            equations_from_matrix(multiplier_array_no_delta[no_delta], var_array)
        )
        results[idx] = np.prod((inverse, no_inverse))
    return results


def equations_with_sympy(multiplier_array, paths, var_array, time, delta=None):
    eqs = np.zeros(len(paths), dtype=object)
    for i, path in enumerate(paths):
        ma = multiplier_array[np.array(path, dtype=int)]
        temp = np.prod(equations_from_matrix(ma, var_array))
        T = sympy.symbols("T", real=True, positive=True)
        eqs[i] = linverse.return_inverse_laplace_sympy(temp, delta, T).subs({T: time})
    return eqs


def make_branchtype_dict_idxs_gimble():
    return {"a": 1, "abb": 1, "b": 0, "aab": 0, "ab": 2, "aa": 3, "bb": 3}


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
    max_value = max(itertools.chain.from_iterable(eq_array))
    to_invert = np.array([t[1][1] for t in sorted_eq_dict])
    dtype = np.min_scalar_type(max_value + 1)
    fill_value = np.iinfo(dtype).max
    adjacency_matrix = np.full((max_idx, max_idx), fill_value=fill_value, dtype=dtype)
    for idx, ((p, c), _) in enumerate(sorted_eq_dict):
        adjacency_matrix[p, c] = idx
    return (adjacency_matrix, eq_array, to_invert)


def make_graph(gfobj):
    stack = [
        (0, gfobj.sample_list),
    ]
    eq_list = list()
    eq_idx = 0
    node_idx = 1
    graph_dict = collections.defaultdict(list)
    equation_dict = dict()  # key=(parent, child), value=eq_idx
    nodes_visited = set()  # set of all nodes visisted
    str_to_numeric_node_dict = {gflib.sample_to_str(gfobj.sample_list): 0}

    while stack:
        parent_node_numeric, state = stack.pop()
        parent_node = gflib.sample_to_str(state)
        if sum(len(pop) for pop in state) > 1:
            if parent_node not in nodes_visited:
                nodes_visited.add(parent_node)
                multiplier_array, new_state_list = gfobj.gf_single_step(state)
                eq_list.append(multiplier_array)
                for _, new_state in zip(multiplier_array, new_state_list):
                    child_node = gflib.sample_to_str(new_state)
                    if child_node in str_to_numeric_node_dict:
                        child_node_numeric = str_to_numeric_node_dict[child_node]
                    else:
                        child_node_numeric = node_idx
                        str_to_numeric_node_dict[child_node] = child_node_numeric
                        node_idx += 1
                    graph_dict[parent_node_numeric].append(child_node_numeric)
                    equation_dict[(parent_node_numeric, child_node_numeric)] = eq_idx
                    stack.append((child_node_numeric, new_state))
                    eq_idx += 1

    graph_array = [
        tuple(graph_dict[i]) if i in graph_dict else tuple() for i in range(node_idx)
    ]
    adjacency_matrix = eq_dict_to_adjacency_matrix(equation_dict, node_idx, eq_idx)

    return (graph_array, adjacency_matrix, np.concatenate(eq_list, axis=0))


def collapse_graph(gfobj, graph_array, adjacency_matrix, eq_matrix):
    num_discrete_events = len(gfobj.discrete_events)

    if num_discrete_events == 0:
        collapsed_graph_array = graph_array
        eq_array = tuple([(i,) for i in range(eq_matrix.shape[0])])
        to_invert_array = np.zeros(len(eq_array), dtype=bool)
    else:
        if num_discrete_events > 1:
            raise NotImplementedError
        delta_idx = gfobj.discrete_events[0]
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
        ) = eq_dict_to_adjacency_matrix_collapsed_graph(equations_dict, new_node_idx)
    return (
        collapsed_graph_array,
        adjacency_matrix,
        eq_array,
        to_invert_array,
    )

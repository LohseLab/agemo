import collections
import itertools

import numpy as np

import agemo.gflib as gflib


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


def coalesce_lineages(input_tuple, to_join):
    """
    Joins lineages and returns resulting tuple of lineages for a single pop
    """
    result = list(input_tuple)
    for lineage in to_join:
        result.remove(lineage)
    result.append("".join(sorted(gflib.flatten(to_join))))
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
            assert migration_direction or exodus_direction, (
                "lineages from different populations cannot coalesce"
                " without migration or exodus event."
            )
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
        current_branches = list(gflib.flatten(state_list))
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
            parent_node = gflib.sample_to_str(state_list)
            if sum(len(pop) for pop in state_list) == 1:
                paths.append(path_so_far)
            else:
                if parent_node in nodes_visited:
                    # depth first search through graph
                    for add_on_path in gflib.paths_from_visited_node(
                        graph_dict, parent_node, equation_dict, path_so_far
                    ):
                        paths.append(add_on_path)
                else:
                    nodes_visited.append(parent_node)
                    for eq, new_state_list in self.gf_single_step(state_list):
                        child_node = gflib.sample_to_str(new_state_list)
                        eq_idx = len(eq_list)
                        eq_list.append(eq)
                        path = path_so_far[:]
                        path.append(eq_idx)
                        graph_dict[parent_node].append(child_node)
                        equation_dict[(parent_node, child_node)] = eq_idx
                        stack.append((path, new_state_list))

        return (paths, np.array(eq_list))

    def gf_single_step(self, state_list):
        current_branches = list(gflib.flatten(state_list))
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

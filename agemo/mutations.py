import itertools
import collections
import copy
import math
import numpy as np
import numba

import agemo.gflib as gflib


def return_mutype_configs(max_k, include_marginals=True):
    add_to_k = 2 if include_marginals else 1
    iterable_range = (range(k + add_to_k) for k in max_k)
    return itertools.product(*iterable_range)


def make_mutype_tree_single_digit(all_mutypes, root, max_k):
    result = collections.defaultdict(list)
    for idx, mutype in enumerate(all_mutypes):
        if mutype == root:
            pass
        else:
            if any(m > max_k_m for m, max_k_m in zip(mutype, max_k)):
                root_config = tuple(
                    m if m <= max_k_m else 0 for m, max_k_m in zip(mutype, max_k)
                )
                result[root_config].append(mutype)
            else:
                root_config = differs_one_digit(mutype, all_mutypes[:idx])
                result[root_config].append(mutype)
    return result


def make_mutype_tree(all_mutypes, root, max_k):
    result = collections.defaultdict(list)
    for idx, mutype in enumerate(all_mutypes):
        if mutype == root:
            pass
        else:
            if any(m > max_k_m for m, max_k_m in zip(mutype, max_k)):
                root_config = tuple(
                    m if m <= max_k_m else 0 for m, max_k_m in zip(mutype, max_k)
                )
                result[root_config].append(mutype)
            else:
                root_config = closest_digit(mutype, all_mutypes[:idx])
                result[root_config].append(mutype)
    return result


def differs_one_digit(query, complete_list):
    # complete_set = self.generate_differs_one_set(query)
    # similar_to = next(obj for obj in complete_list[idx-1::-1] if obj in complete_set)
    similar_to = next(
        obj for obj in complete_list[::-1] if sum_tuple_diff(obj, query) == 1
    )
    return similar_to


def closest_digit(query, complete_list):
    return min(complete_list[::-1], key=lambda x: tuple_distance(query, x))


def tuple_distance(a_tuple, b_tuple):
    dist = [a - b for a, b in zip(a_tuple, b_tuple)]
    if any(x < 0 for x in dist):
        return np.inf
    else:
        return abs(sum(dist))


def sum_tuple_diff(tuple_a, tuple_b):
    return sum(b - a for a, b in zip(tuple_a, tuple_b))


# dealing with marginals
def list_marginal_idxs(marginal, max_k):
    marginal_idxs = np.argwhere(marginal > max_k).reshape(-1)
    shape = np.array(max_k, dtype=np.uint8) + 2
    max_k_zeros = np.zeros(shape, dtype=np.uint8)
    slicing = [
        v if idx not in marginal_idxs else slice(-1)
        for idx, v in enumerate(marginal[:])
    ]
    max_k_zeros[slicing] = 1
    return [tuple(idx) for idx in np.argwhere(max_k_zeros)]


def add_marginals_restrict_to(restrict_to, max_k):
    marginal_np = np.array(restrict_to, dtype=np.uint8)
    marginal_mutypes_idxs = np.argwhere(np.any(marginal_np > max_k, axis=1)).reshape(-1)
    if marginal_mutypes_idxs.size > 0:
        result = []
        # for mut_config_idx in marginal_mutypes_idxs:
        #   print(marginal_np[mut_config_idx])
        #   temp = list_marginal_idxs(marginal_np[mut_config_idx], max_k)
        #   result.append(temp)
        #   print(temp)
        result = [
            list_marginal_idxs(marginal_np[mut_config_idx], max_k)
            for mut_config_idx in marginal_mutypes_idxs
        ]
        result = list(itertools.chain.from_iterable(result)) + restrict_to
        result = sorted(set(result))
    else:
        return sorted(restrict_to)
    return result


@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def max_or_zero_subtraction(x, y):
    if x == 0:
        return np.maximum(0, x - y)
    else:
        return x - y


def adjust_marginals_array(array, dimension):
    new_array = copy.deepcopy(array)  # why the deepcopy here?
    for j in range(dimension):
        new_array = _adjust_marginals_array(new_array, dimension, j)
    return new_array


def _adjust_marginals_array(array, dimension, j):
    idxs = np.roll(range(dimension), j)
    result = array.transpose(idxs)
    # result[-1] = result[-1] - np.sum(result[:-1], axis=0)
    # line below avoids calculations with marginals for impossible mutation types
    result[-1] = np.maximum(
        np.zeros_like(result[-1]), result[-1] - np.sum(result[:-1], axis=0)
    )
    new_idxs = np.zeros(dimension, dtype=np.uint8)
    new_idxs[np.transpose(idxs)] = np.arange(dimension, dtype=np.uint8)
    return result.transpose(new_idxs)


# making branchtype dict


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


def make_branchtype_dict_idxs(
    sample_list, phased=False, rooted=False, starting_index=0
):
    samples = sorted(gflib.flatten(pop for pop in sample_list if len(pop) > 0))
    if phased:
        all_branchtypes = list(
            flatten(
                [
                    ["".join(p) for p in itertools.combinations(samples, i)]
                    for i in range(1, len(samples))
                ]
            )
        )
    else:
        all_branchtypes = list(
            flatten(
                [
                    sorted(
                        set(["".join(p) for p in itertools.combinations(samples, i)])
                    )
                    for i in range(1, len(samples))
                ]
            )
        )

    fold = math.ceil(len(all_branchtypes) / 2)
    correction_idx = len(all_branchtypes) - 1 if rooted else 0
    branchtype_dict = dict()
    for idx in range(fold):
        branchtype_dict[all_branchtypes[idx]] = idx + starting_index
        branchtype_dict[all_branchtypes[-idx - 1]] = (
            abs(-correction_idx + idx) + starting_index
        )

    return branchtype_dict


# initial outline general branchtype implementation
def get_binary_branchtype_array_all(num_samples, phased=False, rooted=False):
    if phased:
        bta = get_binary_branchtype_array(sum(num_samples))  # should be int
    else:
        bta = get_binary_branchtype_array_unphased(num_samples)  # should be tuple
    if not rooted:
        return bta[: math.ceil(bta.shape[0] / 2)]
    return bta


def get_binary_branchtype_array(num_samples):
    all_nums = [
        [p for p in itertools.combinations(range(num_samples), i)]
        for i in range(1, num_samples)
    ]
    t = list(flatten(all_nums))
    x = np.zeros((len(t), num_samples), dtype=np.uint8)
    for idx, subt in enumerate(t):
        x[idx, subt] = 1
    return x


@numba.vectorize([numba.uint8(numba.uint8, numba.uint8)])
def packbits(x, y):
    return 2 * x + y


def binary_to_decimal(b):
    return packbits.reduce(b, axis=-1)


def flatten(input_list):
    # exists somewhere else
    return itertools.chain.from_iterable(input_list)


def stacked_branchtype_representation(idx, binary_btype_array):
    idx = np.array(idx)
    return binary_btype_array[idx == 1]


def ravel_multi_index(multi_index, shape):
    # exists somewhere else
    shape_prod = np.cumprod(shape[:0:-1])[::-1]
    return np.sum(shape_prod * multi_index[:-1]) + multi_index[-1]


# determining binary_branchtype_array for unphased data


def recursive_distribute(a, idx, to_distribute, boundary):
    if to_distribute == 0:
        yield tuple(a)
    else:
        if idx < len(a):
            distribute_now = min(to_distribute, boundary[idx])
            for i in range(distribute_now, -1, -1):
                new_a = a[:]
                new_a[idx] = i
                yield from recursive_distribute(
                    new_a, idx + 1, to_distribute - i, boundary
                )


def generate_idxs(boundary):
    total_boundary = sum(boundary)
    a = [0] * len(boundary)
    for i in range(1, total_boundary):
        yield from recursive_distribute(a, 0, i, boundary)


# from idxs to binary branchtype array
def get_binary_branchtype_array_unphased(num_samples_per_pop):
    total_num_samples = sum(num_samples_per_pop)
    correction = int(max(num_samples_per_pop) == total_num_samples)
    result = []
    for config in generate_idxs(num_samples_per_pop):
        temp = []
        for idx, pop in enumerate(config):
            temp += [1] * pop + [0] * (num_samples_per_pop[idx] - pop)
        result.append(tuple(temp))
    return np.array(result, dtype=np.uint8)[:, : (total_num_samples - correction)]


# testing branchtype compatibility
def branchtype_compatibility(b):
    # returns all incompatible branchtypes for a given array of branchtypes b
    num_rows = b.shape[0]
    incompatible = [set() for _ in range(num_rows)]
    for idx1, idx2 in itertools.combinations(range(num_rows), 2):
        if not intersection_check(b[idx1], b[idx2]):
            incompatible[idx1].add(idx2)
            incompatible[idx2].add(idx1)
    return incompatible


def intersection_check(a, b):
    s1 = set(np.argwhere(a).reshape(-1))
    s2 = set(np.argwhere(b).reshape(-1))
    min_len = min(len(s1), len(s2))
    len_intersection = len(s1.intersection(s2))
    return len_intersection == min_len or len_intersection == 0


def compatibility_depth_first(compatibility_check, size):
    root = ((0,) * size, compatibility_check[0], size - 1)
    stack = [root]
    possible = []
    while stack:
        mutype, incompatibility_set, idx = stack.pop()
        if idx > -1:
            new_mutype = list(mutype)
            new_mutype[idx] = 1
            new_mutype = tuple(new_mutype)
            stack.append((mutype, incompatibility_set, idx - 1))
            if idx not in incompatibility_set:
                new_incompatibility_set = incompatibility_set.union(
                    compatibility_check[idx]
                )
                stack.append((new_mutype, new_incompatibility_set, idx - 1))
        else:
            possible.append(mutype)

    return possible


def all_within_deme_permutations(b, slices=None):
    if slices is None:
        slices = np.array([])
    b_pop = np.split(b, slices, axis=-1)
    for p in all_permutations(*b_pop):
        yield p


def all_permutations(*b):
    for r in itertools.product(*[single_permutation_2d(m) for m in b]):
        yield np.hstack(tuple(r))


def single_permutation_2d(b):
    dim = b.shape[-1]
    if len(np.unique(b)) == 1:
        yield b
        return
    else:
        for ordering in itertools.permutations(range(dim)):
            yield b[:, np.array(ordering)]


def equivalences_single_mutype(
    mutype, mutype_shape, stack_to_idx, branchtype_array, slices
):
    num_mutypes = len(mutype_shape)
    non_zero_idx_base = np.flatnonzero(mutype)
    stacked_repr = branchtype_array[non_zero_idx_base]
    all_permutations = list(all_within_deme_permutations(stacked_repr, slices))
    mutations_to_distribute = np.array(mutype_shape)[non_zero_idx_base]
    non_zero_idxs_equivalent = np.zeros(
        (len(all_permutations) - 1, len(mutations_to_distribute)), dtype=int
    )

    for pidx in range(len(all_permutations) - 1):
        dec_repr = binary_to_decimal(all_permutations[pidx + 1])
        non_zero_idxs_equivalent[pidx] = [stack_to_idx[dec] for dec in dec_repr]

    for d in itertools.product(*(range(1, a) for a in mutations_to_distribute)):
        result = np.zeros(num_mutypes, dtype=int)
        result[non_zero_idx_base] = d
        base_mutype = tuple(result)
        for non_zero_idx_equivalent in non_zero_idxs_equivalent:
            result = np.zeros(num_mutypes, dtype=int)
            result[non_zero_idx_equivalent] = d
            yield (base_mutype, tuple(result))


def distribute_mutations(mutationtype, mutype_shape):
    boolean = np.array(mutationtype).copy().astype(bool)
    result = np.zeros(len(mutationtype), dtype=np.uint8)
    for d in itertools.product(*(range(1, a) for a in mutype_shape[boolean])):
        result[boolean] = d
        yield tuple(result)


def distribute_mutations_all_mutypes(possible_mutationtypes, mutype_shape):
    mutype_shape = np.array(mutype_shape, dtype=np.uint8)
    for mt in possible_mutationtypes:
        yield from distribute_mutations(mt, mutype_shape)


class TypeCounter:
    def __init__(self):
        pass

    @property
    def sample_configuration(self):
        return self._sample_configuration

    @property
    def samples_per_pop(self):
        return self._samples_per_pop

    @property
    def labels(self):
        return self._labels

    @property
    def labels_dict(self):
        return self._labels_dict

    @property
    def binary_representation(self):
        return self._binary_representation

    @property
    def phased(self):
        return self._phased

    @property
    def rooted(self):
        return self._rooted


class BranchTypeCounter(TypeCounter):
    """
    Collects all information on the branch types possible given the `sample_configuration`.

    :param sample_configuration: Should contain a list or tuple for each of the populations involved in the
        coalescent history of the sample, even those that do not contain any lineages at the time of sampling.
        Example: `[(), ("a", "a"), ("b", "b")]`.
    :type sample_configuration: list(list(str))
    :param phased: Indicates whether we can distinguish samples within the same population, defaults to False.
        When False, all samples within the same population get the same identifier. Ideally,
        the `sample_configuration` as provided would already be simplified accordingly.
    :type phased: bool
    :param rooted: Indicates whether branch types should be rooted, defaults to False
    :type rooted: bool
    :param branchtype_dict: Mapping of all branchtypes to an integer index. Changes the order
        of the coefficients within the equation matrix for each of the branch types,
        defaults to `None`
    :type branchtype_dict: dict

    """

    def __init__(
        self,
        sample_configuration,
        phased=False,
        rooted=False,
        branchtype_dict=None,
    ):
        self._sample_configuration = sample_configuration
        self._samples_per_pop = tuple((len(pop) for pop in sample_configuration))
        self._labels, self._labels_dict = self._set_labels(phased, rooted)
        self._binary_representation = get_binary_branchtype_array_all(
            self._samples_per_pop, phased, rooted
        )
        self._custom_mapping = self.custom_branchtype_dict_mapping(branchtype_dict)
        self._compatibility_check = branchtype_compatibility(self.binary_representation)
        self._phased, self._rooted = phased, rooted

    @property
    def compatibility_check(self):
        return self._compatibility_check

    @property
    def custom_mapping(self):
        return self._custom_mapping

    def __len__(self):
        """
        Returns number of branch types.

        """

        return len(self._labels)

    def custom_branchtype_dict_mapping(self, branchtype_dict):
        if branchtype_dict is None:
            return None
        keys_set = set(self.labels_dict.keys())
        custom_keys_set = set(branchtype_dict.keys())
        assert (
            keys_set == custom_keys_set
        ), f'branchtype dict needs to contain all following keys: {", ".join(self.labels)}'
        custom_values_set = set(branchtype_dict.values())
        max_value = max(custom_values_set)
        assert max_value == len(custom_values_set) - 1
        assert min(custom_values_set) == 0
        # sorted_branchtype_array = sorted(
        #    branchtype_dict, key=lambda k: branchtype_dict[k]
        # )
        self._labels_dict = {k: v for k, v in branchtype_dict.items()}
        temp_labels = ["" for _ in range(len(self._labels))]
        for branchtype in self._labels:
            temp_labels[branchtype_dict[branchtype]] = branchtype
        mapping = [self._labels_dict[b] for b in self._labels]
        self._labels = temp_labels

        return np.array(mapping, dtype=np.uint8)

    def _set_labels(self, phased, rooted, starting_index=0):
        # sample_list, phased=False, rooted=False, starting_index=0
        samples = sorted(
            gflib.flatten(pop for pop in self.sample_configuration if len(pop) > 0)
        )
        if phased:
            all_branchtypes = list(
                flatten(
                    [
                        ["".join(p) for p in itertools.combinations(samples, i)]
                        for i in range(1, len(samples))
                    ]
                )
            )
        else:
            all_branchtypes = list(
                flatten(
                    [
                        sorted(
                            set(
                                ["".join(p) for p in itertools.combinations(samples, i)]
                            )
                        )
                        for i in range(1, len(samples))
                    ]
                )
            )

        fold = math.ceil(len(all_branchtypes) / 2)
        correction_idx = len(all_branchtypes) - 1 if rooted else 0
        branchtype_dict = dict()
        for idx in range(fold):
            branchtype_dict[all_branchtypes[idx]] = idx + starting_index
            branchtype_dict[all_branchtypes[-idx - 1]] = (
                abs(-correction_idx + idx) + starting_index
            )

        if rooted:
            return all_branchtypes, branchtype_dict
        else:
            return all_branchtypes[:fold], branchtype_dict


class MutationTypeCounter(TypeCounter):
    """
    Collects all the information on all possible mutation types given a set of branch types.

    :param branch_types: All branchtypes for a given sample set.
    :type branch_types: class `agemo.BranchTypeCounter`
    :param mutype_shape: Tuple of integers of length equal to the number of different branch types.
        Provides the shape for the bSFS. In that case, the shape equates to :math:`k_{max} + 2`. Here
        :math:`k_{max}` indicates the maximum number of mutations along each of the branch types for which
        we want to calculate the likelihood.
    :type mutype_shape: tuple(int)

    """

    def __init__(self, BranchTypeCounter, mutype_shape):
        compatibility_check = branchtype_compatibility(
            BranchTypeCounter.binary_representation
        )
        num_branchtypes = len(BranchTypeCounter)
        self._binary_representation = np.array(
            compatibility_depth_first(compatibility_check, num_branchtypes),
            dtype=np.uint8,
        )

        if BranchTypeCounter.custom_mapping is not None:
            # taking care of custom branchtype_dict
            self._binary_representation = self._binary_representation[
                :, BranchTypeCounter.custom_mapping
            ]

        self._BranchTypeCounter = BranchTypeCounter
        self._mutype_shape = mutype_shape
        (
            self._all_mutypes,
            self._all_mutypes_ravel,
        ) = self.generate_and_sort_all_mutypes()

        self._equiprobable_mutypes_ravel = None  # array if BranchTypeCounter.phased
        self._equiprobable_to_base_mutype = None  # dict if BranchTypeCounter.phased
        self._sample_slicing = np.cumsum(
            [i for i in self._BranchTypeCounter._samples_per_pop if i != 0]
        )[:-1]

    @property
    def mutype_shape(self):
        return self._mutype_shape

    @property
    def all_mutypes(self):
        return self._all_mutypes

    @property
    def all_mutypes_ravel(self):
        return self._all_mutypes_ravel

    @property
    def equiprobable_mutypes_ravel(self):
        return self._equiprobable_mutypes_ravel

    @property
    def BranchTypeCounter(self):
        return self._BranchTypeCounter

    @property
    def sample_configuration(self):
        return self._BranchTypeCounter._sample_configuration

    @property
    def samples_per_pop(self):
        return self._BranchTypeCounter._samples_per_pop

    @property
    def labels(self):
        return self._BranchTypeCounter._labels

    @property
    def labels_dict(self):
        return self._BranchTypeCounter._labels_dict

    @property
    def phased(self):
        return self._BranchTypeCounter._phased

    @property
    def rooted(self):
        return self._BranchTypeCounter._rooted

    def __len__(self):
        """
        Returns number of unique possible mutypes.

        """
        return len(self._all_mutypes)

    def generate_and_sort_all_mutypes(self):
        if self.phased:
            return self.generate_and_sort_all_mutypes_phased()
        else:
            return self.generate_and_sort_all_mutypes_unphased()

    def generate_and_sort_all_mutypes_unphased(self):
        all_mutypes_unsorted = []
        ravel_idxs = []
        for mutype in distribute_mutations_all_mutypes(
            self.binary_representation, self.mutype_shape
        ):
            all_mutypes_unsorted.append(mutype)
            ravel_idxs.append(ravel_multi_index(mutype, self.mutype_shape))
        all_mutypes_unsorted = np.array(all_mutypes_unsorted, dtype=np.int64)
        ravel_idxs = np.array(ravel_idxs, dtype=np.int64)
        ravel_idxs_to_sort = np.argsort(ravel_idxs)
        # return sorted multi-index mutypes, and ravelled version
        return (
            all_mutypes_unsorted[ravel_idxs_to_sort],
            ravel_idxs[ravel_idxs_to_sort],
        )

    def generate_and_sort_all_mutypes_phased(self):
        stack_to_idx = {
            dec: idx
            for idx, dec in enumerate(
                binary_to_decimal(self.BranchTypeCounter.binary_representation)
            )
        }
        self._equiprobable_to_base_mutype = {}
        ravel_idxs, all_mutypes_unsorted = [], []
        for mutype in self._binary_representation:
            ravel_mutype = multi_index_ravel(mutype, self.mutype_shape)
            if ravel_mutype not in self._equiprobable_mutypes_ravel:
                equivalent_mutype_list = []
                for base_mutype, equivalent_mutype in equivalences_single_mutype(
                    mutype,
                    self.mutype_shape,
                    stack_to_idx,
                    self.BranchTypeCounter.binary_representation,
                    self._sample_slicing,
                ):

                    base_mutype_ravel = multi_index_ravel(
                        base_mutype, self.mutype_shape
                    )
                    self._equiprobable_to_base_mutype[
                        multi_index_ravel(equivalent_mutype, self.mutype_shape)
                    ] = base_mutype_ravel
                    ravel_idxs.append(base_mutype_ravel)
                    all_mutypes_unsorted.append(base_mutype)
                    equivalent_mutype_list.append(equivalent_mutype_ravel)

            self._equiprobable_mutypes_ravel.append(
                numpy.array(equivalent_mutype_list, dtype=np.int64)
            )

        ravel_idxs = np.array(ravel_idxs, dtype=np.int64)
        ravel_idxs_to_sort = np.argsort(ravel_idxs)
        temp = [None for _ in range(len(ravel_idxs))]
        for idx in range(len(temp)):
            temp[idx] = self._equiprobable_mutypes_ravel[ravel_idxs_to_sort[idx]]
        self._equiprobable_mutypes_ravel = temp

        return all_mutypes_unsorted[ravel_idxs_to_sort], ravel_idxs[ravel_idxs_to_sort]

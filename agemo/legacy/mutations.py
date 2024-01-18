import itertools

import mpmath
import numpy as np
import sympy

import agemo.gflib as gflib


def single_partial(ordered_mutype_list, partial):
    return list(
        gflib.flatten(
            itertools.repeat(branchtype, count)
            for count, branchtype in zip(partial, ordered_mutype_list)
        )
    )


def make_result_dict_from_mutype_tree_alt(
    gf,
    mutype_tree,
    theta,
    rate_dict,
    ordered_mutype_list,
    max_k,
    precision=165,
):
    root = tuple(0 for _ in max_k)  # root is fixed
    num_mutypes = len(max_k)
    result = np.zeros(max_k + 2, dtype=object)
    stack = [(root, gf)]
    result[root] = eval_equation(gf, theta, rate_dict, root, precision)
    while stack:
        parent, parent_equation = stack.pop()
        if parent in mutype_tree:
            for child in mutype_tree[parent]:
                mucounts = [m for m, max_k_m in zip(child, max_k) if m <= max_k_m]
                marginal = len(mucounts) < num_mutypes
                child_equation = generate_equation(
                    parent_equation,
                    parent,
                    child,
                    max_k,
                    ordered_mutype_list,
                    marginal,
                )
                stack.append((child, child_equation))
                result[child] = eval_equation(
                    child_equation, theta, rate_dict, mucounts, precision
                )
    return result


def generate_equation(equation, parent, node, max_k, ordered_mutype_list, marginal):
    if marginal:
        marginals = {
            branchtype: 0
            for branchtype, count, max_k_m in zip(ordered_mutype_list, node, max_k)
            if count > max_k_m
        }
        return equation.subs(marginals)
    else:
        relative_config = [b - a for a, b in zip(parent, node)]
        partial = single_partial(ordered_mutype_list, relative_config)
        diff = equation.diff(partial)
        return diff


@mpmath.workdps(100)
def eval_equation(derivative, theta, ratedict, numeric_mucounts):
    mucount_total = np.sum(numeric_mucounts)
    mucount_fact_prod = np.prod(
        [np.math.factorial(count) for count in numeric_mucounts]
    )
    # mucount_fact_prod = np.prod(factorials[numeric_mucounts])
    return mpmath.mpf(
        (-1 * theta) ** (mucount_total) / mucount_fact_prod * derivative.subs(ratedict)
    )


# alternative make_result using depth first
def depth_first_mutypes(max_k, labels, eq, theta, rate_dict, exclude=None):
    # factorials = np.cumprod(np.arange(1, np.max(max_k)+1))
    # factorials = np.hstack((1,factorials))
    k = len(max_k) - 1
    stack = [
        (tuple([0 for _ in range(len(max_k))]), k, eq),
    ]
    result = np.zeros(max_k + 2, dtype=np.float64)
    if exclude is None:
        exclude = tuple()
    while stack:
        mutype, k, eq = stack.pop()
        if k > 0:
            for step in single_step_df_mutypes_diff(
                mutype, labels[k], k, max_k[k], eq, theta, exclude
            ):
                stack.append(step)
        else:
            for new_mutype, _, new_eq in single_step_df_mutypes_diff(
                mutype, labels[k], k, max_k[k], eq, theta, exclude
            ):
                mucounts = np.array(
                    [m for m, max_k_m in zip(new_mutype, max_k) if m <= max_k_m]
                )
                temp = eval_equation(new_eq, theta, rate_dict, mucounts)
                result[new_mutype] = temp

    return result


def single_step_df_mutypes_diff(mutype, label, k, max_k, eq, theta, exclude):
    subsdict = {label: theta}
    # for i==0
    yield (mutype, k - 1, eq.subs(subsdict))
    if len(exclude) == 0 or (k != exclude[0] or mutype[exclude[1]] == 0):
        new_eq = eq
        # for i 1 .. max_k
        temp = list(mutype)
        for i in range(1, max_k + 1):
            temp[k] = i
            new_eq = new_eq.diff(label)
            yield (tuple(temp), k - 1, new_eq.subs(subsdict))
        # for i==max_k+1
        subsdict[label] = 0
        temp[k] = max_k + 1
        new_eq = eq.subs(subsdict)
        yield (tuple(temp), k - 1, new_eq)


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


def make_branchtype_dict(sample_list, mapping="unrooted", labels=None):
    """
    Maps lineages to their respective mutation type
    Possible mappings: 'unrooted', 'label'
    """
    all_branchtypes = list(gflib.flatten(sample_list))
    branches = [
        branchtype
        for branchtype in powerset(all_branchtypes)
        if len(branchtype) > 0 and len(branchtype) < len(all_branchtypes)
    ]
    if mapping.startswith("label"):
        if labels:
            assert len(branches) == len(
                labels
            ), "number of labels does not match number of branchtypes"
            branchtype_dict = {
                branchtype: sympy.symbols(label, positive=True, real=True)
                for branchtype, label in zip(branches, labels)
            }
        else:
            branchtype_dict = {
                branchtype: sympy.symbols(f"z_{branchtype}", positive=True, real=True)
                for branchtype in branches
            }
    elif mapping == "unrooted":  # this needs to be extended to the general thing!
        if not labels:
            labels = ["m_1", "m_2", "m_3", "m_4"]
        assert set(all_branchtypes) == {"a", "b"}
        branchtype_dict = dict()
        for branchtype in powerset(all_branchtypes):
            if len(branchtype) == 0 or len(branchtype) == len(all_branchtypes):
                pass
            elif branchtype in ("abb", "a"):
                branchtype_dict[branchtype] = sympy.symbols(
                    labels[1], positive=True, real=True
                )  # hetA
            elif branchtype in ("aab", "b"):
                branchtype_dict[branchtype] = sympy.symbols(
                    labels[0], positive=True, real=True
                )  # hetB
            elif branchtype == "ab":
                branchtype_dict[branchtype] = sympy.symbols(
                    labels[2], positive=True, real=True
                )  # hetAB
            else:
                branchtype_dict[branchtype] = sympy.symbols(
                    labels[3], positive=True, real=True
                )  # fixed difference
    else:
        ValueError("This branchtype mapping has not been implemented yet.")
    return branchtype_dict


# dealing with marginals
def list_marginal_idxs(marginal, max_k):
    marginal_idxs = np.argwhere(marginal > max_k).reshape(-1)
    shape = np.array(max_k, dtype=np.uint8) + 2
    max_k_zeros = np.zeros(shape, dtype=np.uint8)
    # for idx, v in enumerate(marginal[:]):
    slicing = tuple(
        [
            v if idx not in marginal_idxs else slice(-1)
            for idx, v in enumerate(marginal[:])
        ]
    )
    max_k_zeros[slicing] = 1
    return [tuple(idx) for idx in np.argwhere(max_k_zeros)]


def add_marginals_restrict_to(restrict_to, max_k):
    marginal_np = np.array(restrict_to, dtype=np.uint8)
    marginal_mutypes_idxs = np.argwhere(np.any(marginal_np > max_k, axis=1)).reshape(-1)
    if marginal_mutypes_idxs.size > 0:
        # result = []
        result = [
            list_marginal_idxs(marginal_np[mut_config_idx], max_k)
            for mut_config_idx in marginal_mutypes_idxs
        ]
        result = list(itertools.chain.from_iterable(result)) + restrict_to
        result = sorted(set(result))
    else:
        return sorted(restrict_to)
    return result

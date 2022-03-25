import numpy as np


def inverse_laplace_single_event(multiplier_array, var_array, time, delta_in_nom_list):
    """
    Inverse laplace of (F/delta, delta, time) when there are no higher order poles
    potential issue 1: equality of two constants in the denominator:
     3 solutions: keep general expression: use sage to take limit,
     or use sage to take inverse laplace.
    solution1: make product multiplier_array.dot(variable_array): nom/denom ->sage.all.inverse_laplace()
    solution2: replace all denom differences equalling 0 by sage.var and take limit for each one
    of them going to 0.
    solution3: this is a higher order pole, use partial fraction expansion algorithm
    issue 2: currently assuming checking for whether delta is present in equation has been done.
    Function will always return an 'inverse', even if taking inverse should be the function itself.

    :param array multiplier_array: Result of the generating function, with delta_idx column left out
    :param array delta_in_nom_list: boolean array: Describes which factors contain delta in nominator
    :param int delta_idx: index of parameter to take inverse of in multiplier_array
    :param array, list var_array: variable array, with delta left out
    :param float, object time: floatvalue or sage variable
    """
    constants = multiplier_array.dot(var_array)
    constants_denom = constants[:, 1]
    pairwise_differences = constants_denom[:, None] - constants_denom[None, :]
    # sign of lower triangle needs to be adjusted:
    i_upper = np.triu_indices(pairwise_differences.shape[-1], 1)
    pairwise_differences.T[i_upper] = -pairwise_differences.T[
        i_upper
    ]  # adjust sign upper triangle
    i, j = np.diag_indices(pairwise_differences.shape[-1], ndim=2)
    pairwise_differences[i, j] = 1  # adjust diagonal to 1 for product
    denominators = np.prod(pairwise_differences, axis=-1)
    constants_nom = constants[:, 0]
    leading_constants = np.prod(
        constants_nom, initial=1, where=constants_nom != 0, axis=-1
    )
    if any(d == 0 for d in denominators):  # any of denominators entries 0:
        raise ValueError("Inverse Laplace runs into zero values in denominator.")
        # EXCEPTION -> this means there is a higher order pole, can be solved with pfe!
        # poles, multiplicities = np.unique(constants_denom, return_counts=True)
        # if not any(delta_in_nom_list): #no delta in nominator
        # 	poles = np.hstack((poles, 0.0))
        # 	multiplicities = np.hstack((multiplicities, 1))
        # max_multiplicity = np.max(multiplicities)
        # binom_coefficients = gfpfe.return_binom_coefficients(max_multiplicity)
        # factorials = 1/np.cumprod(np.arange(1,max_multiplicity))
        # factorials = np.hstack((1, factorials))
        # return leading_constants * inverse_laplace_PFE(-poles, multiplicities, time, binom_coefficients, factorials, use_numba=False)
    else:
        # REGULAR CASE
        exp_nom = np.exp(-constants_denom * time)
        signs = np.ones(constants_nom.shape[0], dtype=np.int8)
        signs_idxs = np.arange(signs.size, dtype=np.int8)
        signs = np.negative(
            signs, where=signs_idxs % 2 != 0, out=signs
        )  # sign: +/-/+/-/...
        if not any(delta_in_nom_list):  # no delta in nominator
            signs = (-1) ** (len(constants_denom)) * signs
            denominators *= constants_denom
            nominators = leading_constants * signs * exp_nom
            extra_term = leading_constants / np.prod(constants_denom)
            result = nominators / denominators
            return np.concatenate(([extra_term], result))
        else:  # single delta in nominator
            signs = -((-1) ** (len(constants_denom))) * signs
            nominators = leading_constants * signs * exp_nom
            return nominators / denominators

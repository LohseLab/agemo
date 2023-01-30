import itertools
import numba
import numpy as np
import math

# numerical compensation algorithms
# algorithm from Ogita et al. 2005. Accurate sum and dot product. Journal of Scientific Computing
@numba.njit(numba.types.UniTuple(numba.float64, 2)(numba.float64, numba.float64), cache=True)
def two_sum(a, b):
    x = a + b
    y = x - a
    e = a - (x - y) + (b - y)
    return x, e


@numba.njit(numba.float64(numba.float64[:]), cache=True)
def casc_sum(arr):
    s, t = 0, 0
    for x in arr:
        (s, e) = two_sum(s, x)
        t += e
    return s + t


@numba.njit(
    [
        numba.float64(numba.uint8[:], numba.float64[:]),
        numba.float64(numba.uint16[:], numba.float64[:]),
        numba.float64(numba.uint32[:], numba.float64[:]),
        numba.float64(numba.uint64[:], numba.float64[:]),
        numba.float64(numba.int8[:], numba.float64[:]),
        numba.float64(numba.int16[:], numba.float64[:]),
        numba.float64(numba.int32[:], numba.float64[:]),
        numba.float64(numba.int64[:], numba.float64[:]),
    ], cache=True
)
def casc_dot_product(A, B):
    s, t = 0, 0
    for i in range(A.size):
        (s, e) = two_sum(s, A[i] * B[i])
        t += e
    return s + t


# derivatives base functions:
@numba.njit(
    [
        numba.float64(numba.uint8[:], numba.float64[:]),
        numba.float64(numba.uint16[:], numba.float64[:]),
        numba.float64(numba.uint32[:], numba.float64[:]),
        numba.float64(numba.uint64[:], numba.float64[:]),
        numba.float64(numba.int8[:], numba.float64[:]),
        numba.float64(numba.int16[:], numba.float64[:]),
        numba.float64(numba.int32[:], numba.float64[:]),
        numba.float64(numba.int64[:], numba.float64[:]),
    ], cache=True
)
def simple_dot_product(A, B):
    m = A.size
    s = 0
    for i in range(m):
        s += A[i] * B[i]
    return s


# derivatives base functions:
@numba.njit(
    [
        numba.float64(numba.uint8[:], numba.float64[:], numba.int64),
        numba.float64(numba.uint16[:], numba.float64[:], numba.int64),
        numba.float64(numba.uint32[:], numba.float64[:], numba.int64),
        numba.float64(numba.uint64[:], numba.float64[:], numba.int64),
        numba.float64(numba.int8[:], numba.float64[:], numba.int64),
        numba.float64(numba.int16[:], numba.float64[:], numba.int64),
        numba.float64(numba.int32[:], numba.float64[:], numba.int64),
        numba.float64(numba.int64[:], numba.float64[:], numba.int64),
    ], cache=True
)
def simple_dot_product_setback(A, B, setback):
    m = A.size
    s = 0
    for i in range(m - setback):
        s += A[i] * B[i]
    return s


@numba.njit(cache=True)
def taylor_coeff_inverse_polynomial_legacy(
    denom, var_array, diff_array, num_branchtypes, dot_product
):
    # of the form c/f(var_array)
    diff_array = np.array(diff_array, dtype=np.uint8)
    total_diff_count = np.sum(diff_array)
    fact_diff = 1
    for num in diff_array:
        fact_diff *= math.gamma(num + 1)
    fact = math.gamma(total_diff_count + 1) / fact_diff
    nomd = fact * np.prod(denom[-num_branchtypes:] ** diff_array)
    if nomd == 0.0:
        return 0.0
    denomd = dot_product ** (total_diff_count + 1)
    return (-1) ** (total_diff_count) * nomd / denomd


@numba.njit(cache=True)
def taylor_coeff_inverse_polynomial(
    denom, theta, diff_array, num_branchtypes, dot_product, mutypes_shape
):
    # of the form c/f(var_array)
    # diff_array = np.array(diff_array, dtype=np.uint8) # where marg should already be 0!!
    total_diff_count, fact_diff, nomd = 0, 1, 1
    for idx in range(num_branchtypes, 0, -1):
        diff_value = diff_array[-idx] % (mutypes_shape[-idx] - 1)
        denom_marg_corr = denom[-idx] * int(diff_value == diff_array[-idx])
        nomd *= denom_marg_corr**diff_value
        dot_product += theta * denom_marg_corr
        fact_diff *= math.gamma(diff_value + 1)
        total_diff_count += diff_value
    nomd *= math.gamma(total_diff_count + 1) / fact_diff
    if dot_product == 0:
        raise ZeroDivisionError
    if nomd == 0.0:
        return 0.0
    denomd = dot_product ** (total_diff_count + 1)
    return (-1) ** (total_diff_count) * nomd / denomd


@numba.njit(cache=True)
def taylor_coeff_exponential_legacy(
    c, f, exponential_part, diff_array, num_branchtypes
):
    # of the form e**(c*f(var_array))
    # degree of f max 1 for each var
    diff_array = np.array(diff_array, dtype=np.uint8)
    p1 = c ** (np.sum(diff_array)) * np.prod(f[-num_branchtypes:] ** diff_array)
    fact = 1
    for num in diff_array:
        fact *= math.gamma(num + 1)
    return p1 * exponential_part / fact


@numba.njit(cache=True)
def taylor_coeff_exponential(
    c, f, dot_product, diff_array, num_branchtypes, theta, mutypes_shape
):
    # of the form e**(c*f(var_array))
    # degree of f max 1 for each var
    # diff_array = np.array(diff_array, dtype=np.uint8) #make sure this can be omitted!
    p1, fact, sum_diff_array = 1, 1, 0
    for idx in range(num_branchtypes, 0, -1):
        diff_value = diff_array[-idx] % (mutypes_shape[-idx] - 1)
        denom_marg_corr = f[-idx] * int(diff_value == diff_array[-idx])
        p1 *= denom_marg_corr**diff_value
        dot_product += theta * denom_marg_corr
        fact *= math.gamma(diff_value + 1)
        sum_diff_array += diff_value
    p1 *= c**sum_diff_array
    exponential_part = np.exp(c * dot_product)
    return p1 * exponential_part / fact


# combining taylor series
@numba.njit(cache=True)
def series_product_legacy(arr1, arr2, subsetdict):
    # arr1*arr2
    shape = arr1.shape
    result = np.zeros_like(arr1)
    ravel_arr1 = np.ravel(arr1)
    ravel_arr2 = np.ravel(arr2)
    for k in np.ndindex(shape):
        new_idxs = subsetdict[k]
        # result[k] = np.sum(ravel_arr1[new_idxs]*(ravel_arr2[new_idxs][::-1]))
        result[k] = casc_sum(ravel_arr1[new_idxs] * (ravel_arr2[new_idxs][::-1]))
    return result


@numba.njit(cache=True)
def series_product(arr1, arr2, subsetdict):
    # arr1*arr2
    size = arr1.size
    result = np.zeros_like(arr1)
    for k in range(size):
        new_idxs = subsetdict[k]
        result[k] = casc_sum(arr1[new_idxs] * (arr2[new_idxs][::-1]))
    return result


@numba.njit(cache=True)
def series_quotient_legacy(arr1, arr2, subsetdict):
    # arr1/arr2
    shape = arr1.shape
    result = np.zeros_like(arr1)
    # ravel_arr1 = np.ravel(arr1)
    ravel_arr2 = np.ravel(arr2)
    quot = 1 / ravel_arr2[0]
    for k in np.ndindex(shape):
        new_idxs = subsetdict[k]
        result[k] = quot * (
            arr1[k] - np.sum(np.ravel(result)[new_idxs] * ravel_arr2[new_idxs][::-1])
        )
    return result


# making subsetdict
@numba.jit(
    [
        numba.int64(numba.int64[:], numba.int64[:]),
        numba.int64(numba.uint64[:], numba.uint64[:]),
    ], cache=True
)
def ravel_multi_index(multi_index, shape):
    shape_prod = np.cumprod(shape[:0:-1])[::-1]
    return np.sum(shape_prod * multi_index[:-1]) + multi_index[-1]


def return_smaller_than_idx(idx, shape):
    all_indices = itertools.product(*(range(i + 1) for i in idx))
    return np.array(
        [
            ravel_multi_index(np.array(idx2, dtype=int), np.array(shape, dtype=int))
            for idx2 in all_indices
        ],
        dtype=int,
    )


def return_strictly_smaller_than_idx(idx, shape):
    all_indices = itertools.product(*(range(i + 1) for i in idx))
    return np.array(
        [
            ravel_multi_index(np.array(idx2, dtype=int), np.array(shape, dtype=int))
            for idx2 in itertools.takewhile(
                lambda x: x != idx or sum(x) != 0, all_indices
            )
        ],
        dtype=int,
    )


# making subsetdict with marginals
@numba.njit(cache=True)
def increment_marginal(arr, idx, max_value, reset_value):
    if idx < 0:
        return -1
    i = arr[idx]
    i += 1
    if i > max_value[idx]:
        arr[idx] = reset_value[idx]
        idx -= 1
        result = increment_marginal(arr, idx, max_value, reset_value)
    else:
        result = 0
        arr[idx] = i
    return result


@numba.njit(cache=True)
def return_smaller_than_idx_marg(start, max_value, shape):
    reset_value = start.copy()
    yield ravel_multi_index(start, shape)
    check = increment_marginal(start, len(shape) - 1, max_value, reset_value)
    while check == 0:
        yield ravel_multi_index(start, shape)
        check = increment_marginal(start, len(shape) - 1, max_value, reset_value)


def product_subsetdict_marg_legacy(shape):
    # def product_subsetdict_marg(shape, all_idxs)
    # small modification: only make subsetdict for provided all_idxs.
    result = numba.typed.Dict()
    for idx in np.ndindex(shape):
        reset_value = np.zeros(len(shape), dtype=np.int64)
        temp_size = 1
        for i, v in enumerate(idx):
            if v == shape[i] - 1:
                reset_value[i] = v
            else:
                temp_size *= idx[i] + 1
        result[idx] = np.zeros(temp_size, dtype=np.int64)
        i = 0
        for r in return_smaller_than_idx_marg(
            reset_value, idx, np.array(shape, dtype=np.int64)
        ):
            result[idx][i] = r
            i += 1
        assert i == temp_size

    return result


def product_subsetdict_marg(shape, all_mutypes):
    result = numba.typed.Dict()
    for mutype in all_mutypes:
        reset_value = np.zeros(len(shape), dtype=np.uint64)
        temp_size = 1
        for i, v in enumerate(mutype):
            if v == shape[i] - 1:
                reset_value[i] = v
            else:
                temp_size *= mutype[i] + 1
        temp = np.zeros(temp_size, dtype=np.int64)

        for i, r in enumerate(
            return_smaller_than_idx_marg(
                reset_value, mutype, np.array(shape, dtype=np.uint64)
            )
        ):
            temp[i] = r
        result[temp[-1]] = temp
    return result


def product_subsetdict(shape):
    if len(shape) == 0:
        shape = (1,)
    nd = numba.typed.Dict()
    for idx in np.ndindex(tuple(shape)):
        nd[idx] = return_smaller_than_idx(idx, shape)
    return nd


def quotient_subsetdict(shape):
    ndq = numba.typed.Dict()
    for idx in np.ndindex(tuple(shape)):
        ndq[idx] = return_strictly_smaller_than_idx(idx, shape)
    return ndq


# deconstructing equations:
@numba.njit(cache=True)
def quotient_f_g(subsetdict, f, g):
    result = np.zeros_like(f)
    for idx, (fs, gs) in enumerate(zip(f, g)):
        result[idx] = series_quotient_legacy(fs, gs, subsetdict)
    return result


@numba.njit(cache=True)
def product_f(subsetdict, f):
    if len(f) == 1:
        return f[0]
    else:
        result = f[0]
        for f2 in f[1:]:
            result = series_product(result, f2, subsetdict)
        return result


@numba.njit(cache=True)
def product_f_g(subsetdict, f, g, signs):
    if g.shape[0] == 0:
        return signs * f
    else:
        result = np.zeros_like(f)
        for idx, (fs, gs, sign) in enumerate(zip(f, g, signs)):
            result[idx] = sign * series_product(fs, gs, subsetdict)
        return result


@numba.njit(cache=True)
def all_polynomials_legacy(eq_matrix, shape, var_array, num_branchtypes, mutype_shape):
    num_equations = eq_matrix.shape[0]
    if num_equations == 0:
        result = np.zeros((1, *shape), dtype=np.float64)
        result.flat[0] = 1.0
        return result
    else:
        result = np.zeros((num_equations, *shape), dtype=np.float64)
        for idx, eq in enumerate(eq_matrix):
            dot_product = simple_dot_product(eq, var_array)
            if dot_product == 0:
                raise ZeroDivisionError
            for idx2, mutype in zip(np.ndindex(shape), np.ndindex(mutype_shape)):
                # mutype = np.array(mutype, dtype=np.uint8)
                result[(idx, *idx2)] = taylor_coeff_inverse_polynomial_legacy(
                    eq, var_array, mutype, num_branchtypes, dot_product
                )
        return result


def all_polynomials(
    eq_matrix, size, var_array, num_branchtypes, mutype_array, mutype_shape
):
    num_equations = eq_matrix.shape[0]
    if num_equations == 0:
        # return np.array([], dtype=np.float64)
        return np.zeros((num_equations, size), dtype=np.float64)
    else:
        result = np.zeros((num_equations, size))
        theta = var_array[-1]
        for idx, eq in enumerate(eq_matrix):
            dot_product = simple_dot_product_setback(eq, var_array, num_branchtypes)
            # issue here: this is not entire dot product!!!! needs adapting to catch the
            # correct zerodivision errors!!
            # if dot_product==0:
            # 	raise ZeroDivisionError
            for idx2, mutype in enumerate(mutype_array):
                mutype = mutype_array[idx2]
                result[idx, idx2] = taylor_coeff_inverse_polynomial(
                    eq,
                    theta,
                    mutype,
                    num_branchtypes,
                    dot_product,
                    mutype_shape,
                )
        return result


@numba.njit(cache=True)
def all_exponentials_legacy(
    eq_matrix, shape, var_array, time, num_branchtypes, mutype_shape
):
    # eq_matrix contains only denominators!
    num_equations = eq_matrix.shape[0]
    result = np.zeros((num_equations, *shape), dtype=np.float64)
    for idx, eq in enumerate(eq_matrix):
        exponential_part = np.exp(-time * simple_dot_product(eq, var_array))
        # exponential_part = np.exp(-time*eq.dot(var_array))
        for idx2, mutype in zip(np.ndindex(shape), np.ndindex(mutype_shape)):
            # mutype = np.array(mutype, dtype=np.uint8)
            result[(idx, *idx2)] = taylor_coeff_exponential_legacy(
                -time, eq, exponential_part, mutype, num_branchtypes
            )
    return result


def all_exponentials(
    eq_matrix, size, var_array, time, num_branchtypes, mutypes, mutype_shape
):
    # eq_matrix contains only denominators!
    num_equations = eq_matrix.shape[0]
    result = np.zeros((num_equations, size), dtype=np.float64)
    theta = var_array[-1]
    for idx, eq in enumerate(eq_matrix):
        # exponential_part = np.exp(-time*simple_dot_product(eq, var_array))
        dot_product = simple_dot_product_setback(eq, var_array, num_branchtypes)
        for idx2, mutype in enumerate(mutypes):
            result[idx, idx2] = taylor_coeff_exponential(
                -time,
                eq,
                dot_product,
                mutype,
                num_branchtypes,
                theta,
                mutype_shape,
            )
    return result


@numba.njit(cache=True)
def product_pairwise_diff_inverse_polynomial(polynomial_f, shape, combos, subsetdict):
    if shape[0] <= 1:
        return polynomial_f
    else:
        assert polynomial_f.ndim >= 2
        result = np.zeros(shape, dtype=np.float64)
        for idx, combo in enumerate(combos):
            # combo needs to be a np.array
            result[idx] = product_f(subsetdict, polynomial_f[combo])
        return result


def eq_matrix_subtract(eq_matrix):
    eq_matrix = eq_matrix.astype(np.int64)
    n, num_variables = eq_matrix.shape
    num_comparisons = n * (n - 1) // 2
    result = np.zeros((num_comparisons, num_variables), dtype=eq_matrix.dtype)
    for idx, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        result[idx] = eq_matrix[i] - eq_matrix[j]
    return result


def generate_pairwise_idxs(num_equations):
    if num_equations == 0:
        return np.array([], dtype=np.uint8)
    elif num_equations == 1:
        return np.array([[0]], dtype=np.uint8)
    else:
        temp = np.zeros((num_equations, num_equations), dtype=np.uint8)
        n = num_equations * (num_equations - 1) // 2
        temp[np.triu_indices(num_equations, k=1)] = np.arange(n)
        temp = np.tril(temp.T) + np.triu(temp, 1)
        return temp[~np.eye(num_equations, dtype=bool)].reshape((num_equations, -1))


def compile_inverted_eq_legacy(
    eq_matrix, shape, subsetdict, delta_in_nom, mutype_shape
):
    # delta_column should already have been removed from eq_matrix
    # note we need to add np.zeros(eq_matrix.shape[-1], dtype=int)
    # to eq_matrix if no delta in numerator
    # polyf poles should be pairwise differences n*(n-1)/2 for n poles
    # eq_matrix = eq_matrix.astype(np.float64) #required for numba dot product
    num_branchtypes = len(mutype_shape)
    numerators_eq = eq_matrix[:, 0]
    denominators_eq = eq_matrix[:, 1]
    num_equations = len(numerators_eq)
    signs = (-np.ones(num_equations, dtype=np.int8)) ** np.arange(num_equations)
    if not delta_in_nom:
        signs *= (-1) ** (num_equations)
        denominators_eq = np.vstack(
            (denominators_eq, np.zeros_like(denominators_eq[-1]))
        )
    else:
        signs *= -((-1) ** (num_equations))
    eq_matrix_diffs = eq_matrix_subtract(denominators_eq)
    num_terms = num_equations if delta_in_nom else num_equations + 1
    pairwise_idxs = generate_pairwise_idxs(num_terms)

    def _make_eq(var, time):
        constants = numerators_eq.dot(var)
        leading_constant = np.prod(constants[constants > 0])
        # make derivative matrix for all pairwise differences eq_matrix
        try:
            polyf = all_polynomials_legacy(
                eq_matrix_diffs, shape, var, num_branchtypes, mutype_shape
            )
            # combine derivative matrix results
            denoms = product_pairwise_diff_inverse_polynomial(
                polyf, (num_terms, *shape), pairwise_idxs, subsetdict
            )
            # n terms, both expf and denoms should be of length n
            expf = all_exponentials_legacy(
                denominators_eq,
                shape,
                var,
                time,
                num_branchtypes,
                mutype_shape,
            )
            # adapt with signs! diff dims!
            terms = product_f_g(subsetdict, expf, denoms[:num_equations], signs)
            all_terms = np.sum(terms, axis=0)
            if not delta_in_nom:
                all_terms += denoms[-1]
        except ZeroDivisionError:
            raise ZeroDivisionError
        return leading_constant * all_terms

    return _make_eq


def compile_inverted_eq(
    eq_matrix, size, subsetdict, delta_in_nom, mutype_array, mutype_shape
):
    # delta_column should already have been removed from eq_matrix
    # note we need to add np.zeros(eq_matrix.shape[-1], dtype=int)
    # to eq_matrix if no delta in numerator
    # polyf poles should be pairwise differences n*(n-1)/2 for n poles
    # eq_matrix = eq_matrix.astype(np.float64) #required for numba dot product
    num_branchtypes = len(mutype_shape)
    numerators_eq = eq_matrix[:, 0]
    denominators_eq = eq_matrix[:, 1]
    num_equations = len(numerators_eq)
    signs = (-np.ones(num_equations, dtype=np.int8)) ** np.arange(num_equations)
    if not delta_in_nom:
        signs *= (-1) ** (num_equations)
        denominators_eq = np.vstack(
            (denominators_eq, np.zeros_like(denominators_eq[-1]))
        )
    else:
        signs *= -((-1) ** (num_equations))
    eq_matrix_diffs = eq_matrix_subtract(denominators_eq)
    num_terms = num_equations if delta_in_nom else num_equations + 1
    pairwise_idxs = generate_pairwise_idxs(num_terms)

    def _make_eq(var, time):
        constants = numerators_eq.dot(var)
        leading_constant = np.prod(constants[constants > 0])
        # make derivative matrix for all pairwise differences eq_matrix
        try:
            polyf = all_polynomials(
                eq_matrix_diffs,
                size,
                var,
                num_branchtypes,
                mutype_array,
                mutype_shape,
            )
            # combine derivative matrix results
            denoms = product_pairwise_diff_inverse_polynomial(
                polyf, (num_terms, size), pairwise_idxs, subsetdict
            )
            # n terms, both expf and denoms should be of length n
            expf = all_exponentials(
                denominators_eq,
                size,
                var,
                time,
                num_branchtypes,
                mutype_array,
                mutype_shape,
            )
            # adapt with signs! diff dims!
            # have product_f_g return expf if denoms[:num_equations] ...
            terms = product_f_g(subsetdict, expf, denoms[:num_equations], signs)
            all_terms = np.sum(terms, axis=0)
            if not delta_in_nom:
                all_terms += denoms[-1]
        except ZeroDivisionError:
            raise ZeroDivisionError
        return leading_constant * all_terms

    return _make_eq


def compile_non_inverted_eq_legacy(eq_matrix, shape, mutype_shape):
    num_branchtypes = len(mutype_shape)
    # eq_matrix = eq_matrix.astype(np.float64) #required for numba dot product

    def _make_eq(var):
        if eq_matrix.shape[0] == 0:
            return np.zeros((0, *shape), dtype=np.float64)
        constants = eq_matrix[:, 0].dot(var)
        result = all_polynomials_legacy(
            eq_matrix[:, 1], shape, var, num_branchtypes, mutype_shape
        )
        transpose = result.T
        transpose *= constants
        return result

    return _make_eq


def compile_non_inverted_eq(eq_matrix, size, mutype_array, mutype_shape):
    num_branchtypes = len(mutype_shape)

    def _make_eq(var):
        result = np.zeros(size, np.float64)
        constants = eq_matrix[:, 0].dot(var)
        result = all_polynomials(
            eq_matrix[:, 1],
            size,
            var,
            num_branchtypes,
            mutype_array,
            mutype_shape,
        )
        transpose = result.T
        transpose *= constants
        return result

    return _make_eq


def prepare_graph_evaluation_legacy(
    eq_matrix,
    to_invert_array,
    eq_array,
    shape,
    delta_idx,
    subsetdict,
    mutype_shape,
):
    # generates function for each node of the equation graph
    if delta_idx is None:
        eq_matrix_no_delta = eq_matrix
    else:
        eq_matrix_no_delta = np.delete(eq_matrix, delta_idx, axis=2)
    not_to_invert = np.zeros(np.sum(~to_invert_array), dtype=int)
    i = 0
    while i < to_invert_array.size and not to_invert_array[i]:
        not_to_invert[i] = eq_array[i][0]
        i += 1

    f_non_inverted = compile_non_inverted_eq_legacy(
        eq_matrix_no_delta[not_to_invert], shape, mutype_shape
    )
    f_inverted = []
    for idx in range(i, len(to_invert_array)):
        eq_idxs = eq_array[idx]
        eq_idxs = np.array(eq_idxs, dtype=int)
        delta_in_nom = np.any(eq_matrix[eq_idxs][:, 0, delta_idx]) == 1
        f_inverted.append(
            compile_inverted_eq_legacy(
                eq_matrix_no_delta[eq_idxs],
                shape,
                subsetdict,
                delta_in_nom,
                mutype_shape,
            )
        )
    return (f_non_inverted, f_inverted)


def prepare_graph_evaluation_with_marginals_legacy(
    eq_matrix, to_invert_array, eq_array, marg_iterator, delta_idx
):
    # goal: generate (f_non_inverted, f_inverted) pairs for all needed shapes/eq_matrices
    # eq_matrix needs to be prepared in a way that allows us to study marginals
    # marg_boolean?
    # prepare shape and subsetdict (linked to shape)
    all_fs = []
    num_branchtypes = len(marg_iterator[0][0])
    for marg_bool, shape, mutype_shape, subsetdict, _ in zip(*marg_iterator):
        # change eq_matrix appropriately
        # setting branchtypes to 0 that need to be set to 0
        eqm = eq_matrix.copy()
        eqm[..., -num_branchtypes:][..., marg_bool] = 0
        all_fs.append(
            prepare_graph_evaluation_legacy(
                eqm,
                to_invert_array,
                eq_array,
                shape,
                delta_idx,
                subsetdict,
                mutype_shape,
            )
        )
    return all_fs


def prepare_graph_evaluation_with_marginals(
    eq_matrix,
    to_invert_array,
    eq_array,
    size,
    delta_idx,
    subsetdict,
    mutype_array,
    mutype_shape,
):
    # generates function for each node of the equation graph
    if delta_idx is None:
        eq_matrix_no_delta = eq_matrix
    else:
        eq_matrix_no_delta = np.delete(eq_matrix, delta_idx, axis=2)
    # to_invert_array: contains first all potential False, then potential True
    not_to_invert = np.zeros(np.sum(~to_invert_array), dtype=int)
    i = 0
    while i < to_invert_array.size and not to_invert_array[i]:
        not_to_invert[i] = eq_array[i][0]  # idxs of the equations within eq_matrix
        i += 1

    f_non_inverted = compile_non_inverted_eq(
        eq_matrix_no_delta[not_to_invert], size, mutype_array, mutype_shape
    )
    f_inverted = []
    for idx in range(i, len(to_invert_array)):
        eq_idxs = eq_array[idx]
        eq_idxs = np.array(eq_idxs, dtype=int)
        delta_in_nom = np.any(eq_matrix[eq_idxs][:, 0, delta_idx]) == 1
        f_inverted.append(
            compile_inverted_eq(
                eq_matrix_no_delta[eq_idxs],
                size,
                subsetdict,
                delta_in_nom,
                mutype_array,
                mutype_shape,
            )
        )
    return (f_non_inverted, f_inverted)


def evaluate_single_point_with_marginals(size, num_eq_tuple, f_tuple):
    # evaluates single point in parameter space
    # result will be of shape (num_nodes, shape)
    f_non_inverted, f_inverted = f_tuple
    num_eq_non_inverted, num_eq_inverted = num_eq_tuple
    num_eq = num_eq_non_inverted + num_eq_inverted

    def _eval_single_point(var, time):
        result = np.zeros((num_eq, size), dtype=np.float64)
        result[:num_eq_non_inverted] = f_non_inverted(var)
        for idx, f in enumerate(f_inverted):
            result[num_eq_non_inverted + idx] = f(var, time)
        return result

    return _eval_single_point


def evaluate_single_point(shape, f_non_inverted, *f_inverted):
    # evaluates single point in parameter space
    # result will be of shape (num_nodes, shape)
    def _eval_single_point(var, time):
        eval_f_non_inverted = f_non_inverted(var)
        eval_inverted = np.zeros((len(f_inverted), *shape), dtype=np.float64)
        for idx, f in enumerate(f_inverted):
            eval_inverted[idx] = f(var, time)
        result = np.concatenate((eval_f_non_inverted, eval_inverted), axis=0)
        return result

    return _eval_single_point


def evaluate_single_point_with_marginals_legacy(k_max, f_array, num_eq_tuple, slices):
    # f_array should be of shape: (k_max.size**2, 2)
    result_shape = k_max + 2
    num_eq_non_inverted, num_eq_inverted = num_eq_tuple
    num_eq = num_eq_non_inverted + num_eq_inverted

    def _eval_single_point(var, time):
        result = np.zeros((num_eq, *result_shape), dtype=np.float64)
        for ((f_non_inverted, f_inverted), slc) in zip(f_array, slices):
            result[:num_eq_non_inverted][(..., *slc)].flat = f_non_inverted(var).flat
            for idx, f in enumerate(f_inverted):
                result[num_eq_non_inverted + idx][(..., *slc)].flat = f(var, time).flat
        return result

    return _eval_single_point


def marginals_nuissance_objects(k_max):  # legacy
    # all transforms of the same thing, can be improved!
    marg_boolean = generate_booleans(k_max)
    # use marg_bool as diff[marg_bool]=0
    shapes = list(generate_shapes(k_max, marg_boolean))
    slices = list(generate_slices(k_max, marg_boolean))
    subsetdicts = [product_subsetdict(shape) for shape in shapes]
    mutype_shapes = list(generate_mutype_shapes(k_max, marg_boolean, shapes))
    return (marg_boolean, shapes, mutype_shapes, subsetdicts, slices)


def generate_booleans(k_max):  # legacy
    # False = no marginal, True = marginal
    num_repeats = len(k_max)
    return np.array(
        [a for a in itertools.product(range(2), repeat=num_repeats)],
        dtype=bool,
    )


def generate_shapes(k_max, marg_boolean):  # legacy
    # shape with marginal dimensions dropped
    for mb in ~marg_boolean:
        result = tuple(k_max[mb] + 1)
        if len(result) == 0:
            yield (1,)
        else:
            yield result


def generate_mutype_shapes(k_max, marg_boolean, shapes):  # legacy
    # mutype_shape is 1 when marginal, else k_max+1
    num_branchtypes = len(k_max)
    for mb, shape in zip(~marg_boolean, shapes):
        mutype_shape = np.ones(num_branchtypes, dtype=int)
        mutype_shape[mb] = shape
        yield tuple(mutype_shape)


def generate_slices(k_max, marg_boolean):  # legacy
    # generate slices according to provided boolean iter
    for b in marg_boolean:
        yield tuple(
            [
                slice(0, k_max[i] + 1) if not bi else slice(k_max[i] + 1, k_max[i] + 2)
                for i, bi in enumerate(b)
            ]
        )


def taylor_to_probability_coeffs_legacy(shape, include_marginals=False):
    if not include_marginals:
        temp = np.zeros(shape, dtype=np.uint8)
        for idx in np.ndindex(shape):
            temp[idx] = np.sum(idx)
    else:
        mod_factor = np.array(shape)
        shape = mod_factor + 1
        temp = np.zeros(shape, dtype=np.uint8)
        for idx in np.ndindex(temp.shape):
            idx_marginals = np.mod(np.array(idx), mod_factor)
            temp[idx] = np.sum(idx_marginals)
    return temp


def taylor_to_probability_coeffs(mutype_array, mutype_shape, include_marginals=False):
    if not include_marginals:
        return np.sum(mutype_array, axis=-1)
    else:
        size = mutype_array.shape[0]
        mod_factor = np.array(mutype_shape) - 1
        temp = np.zeros(size, dtype=np.uint8)
        for idx, mutype in enumerate(mutype_array):
            idx_marginals = np.mod(mutype, mod_factor)
            temp[idx] = np.sum(idx_marginals)
        return temp


@numba.njit(cache=True)
def taylor_to_probability(precomp, theta):
    max_idx = np.max(precomp) + 1
    temp = np.zeros(max_idx, dtype=np.float64)
    power = 0
    for idx in range(max_idx):
        temp[idx] = (-1 * theta) ** power
        power += 1
    shape = precomp.shape
    result = np.zeros(shape, dtype=np.float64)
    for idx in np.ndindex(shape):
        result[idx] = temp[precomp[idx]]
    return result


def paths(graph, adjacency_matrix):
    stack = [(0, [])]
    visited = []
    while stack:
        parent, path_so_far = stack.pop()
        children = graph[parent]
        if len(children) > 0:
            for child in children:
                path = path_so_far[:]
                path.append(adjacency_matrix[parent, child])
                stack.append((child, path))
        else:
            visited.append(path_so_far)
    return visited


def naive_graph_iterator(evaluated_eqs, paths, subsetdict):
    shape_single = evaluated_eqs[0].shape
    result = np.zeros(shape_single, dtype=np.float64)
    for path in paths:
        # subset_evaluated_eqs = evaluated_eqs[path]
        result += product_f(subsetdict, evaluated_eqs)

    return result


def sort_util(n, visited, stack, graph):
    visited[n] = 1
    children = graph[n]
    for child in children:
        if visited[child] == 0:
            sort_util(child, visited, stack, graph)
    stack.append(n)


def resolve_dependencies(graph):
    visited = np.zeros(len(graph) + 1, dtype=int)
    stack = []
    for idx, node in enumerate(graph):
        if visited[idx] == 0:
            sort_util(idx, visited, stack, graph)
    return np.array(stack, dtype=int)


def iterate_graph(sequence, graph, adjacency_matrix, evaluated_eqs, subsetdict):
    shape = evaluated_eqs[0].shape
    num_nodes = len(sequence)
    node_values = np.zeros((num_nodes, *shape), np.float64)
    for parent in sequence:
        children = graph[parent]
        temp = np.zeros(shape, dtype=np.float64)
        for child in children:
            eq_idx = adjacency_matrix[parent, child]
            if np.any(node_values[child]):
                temp += series_product(
                    node_values[child], evaluated_eqs[eq_idx], subsetdict
                )
            else:
                temp += evaluated_eqs[eq_idx]
        node_values[parent] = temp

    return node_values


@numba.njit(cache=True)
def iterate_eq_graph_legacy(sequence, graph, evaluated_eqs, subsetdict):
    shape = evaluated_eqs[0].shape
    num_nodes = len(sequence)
    node_values = np.zeros((num_nodes, *shape), dtype=np.float64)
    node_values[1:] = evaluated_eqs
    for parent in sequence:
        children = graph[parent]
        temp = np.zeros(shape, dtype=np.float64)
        if len(children) > 0:
            for child in children:
                temp += node_values[child]
            if parent != 0:
                node_values[parent] = series_product_legacy(
                    temp, evaluated_eqs[parent - 1], subsetdict
                )
            else:
                node_values[parent] = temp
    return node_values[0]


@numba.njit(cache=True)
def iterate_eq_graph(sequence, graph, evaluated_eqs, subsetdict):
    size = len(evaluated_eqs[0])
    num_nodes = len(sequence)
    node_values = np.zeros((num_nodes, size), dtype=np.float64)
    node_values[1:] = evaluated_eqs
    for parent in sequence:
        children = graph[parent]
        temp = np.zeros(size, dtype=np.float64)
        if len(children) > 0:
            for child in children:
                temp += node_values[child]
            if parent != 0:
                node_values[parent] = series_product(
                    temp, evaluated_eqs[parent - 1], subsetdict
                )
            else:
                node_values[parent] = temp
    return node_values[0]


def iterate_eq_graph_with_marginals(
    sequence, graph, evaluated_eqs, subsetdicts, slices, shapes, result_shape
):
    # legacy
    num_evaluated_eqs = evaluated_eqs.shape[0]
    result = np.zeros(result_shape, dtype=np.float64)
    for subsetdict, slc, shape in zip(subsetdicts, slices, shapes):
        eval_eq_subset = evaluated_eqs[(..., *slc)].view()
        eval_eq_subset.shape = (num_evaluated_eqs, *shape)
        result[slc].flat = iterate_eq_graph(
            sequence, graph, eval_eq_subset, subsetdict
        ).flat
    return result

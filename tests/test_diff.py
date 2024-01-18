import itertools

import numba
import numpy as np
import pytest
import sympy
from scipy.special import factorial

import agemo.diff as gfdiff
import agemo.legacy.inverse as linverse
import tests.gfdev as gfdev


class TestDotProduct:
    def test_casc_dot_product(self):
        rng = np.random.default_rng()
        A = rng.random(20)
        B = rng.random(20)
        assert np.isclose(
            gfdiff.casc_dot_product(A, B), gfdiff.simple_dot_product(A, B)
        )


@pytest.mark.diff
@pytest.mark.taylor
class TestTaylor:
    def test_taylor_polynomial(self):
        denom = np.arange(1, 5, dtype=int)
        var_array = np.array([1.0, 2.0, 5.0, 5.0], dtype=float) / 10
        diff_array = (2, 2)
        num_mutypes = 2
        dot_product = gfdiff.simple_dot_product_setback(denom, var_array, 2)
        result = gfdiff.taylor_coeff_inverse_polynomial(
            denom, var_array[-1], diff_array, num_mutypes, dot_product, (4, 4)
        )
        expected = 0.84375
        assert np.isclose(expected, result)

    def test_taylor_polynomial_marginals(self):
        denom = np.arange(1, 5, dtype=int)
        var_array = np.array([1.0, 2.0, 5.0, 5.0], dtype=float) / 10
        diff_array = (2, 2)
        num_mutypes = 2
        dot_product = gfdiff.simple_dot_product_setback(denom, var_array, 2)
        result = gfdiff.taylor_coeff_inverse_polynomial(
            denom, var_array[-1], diff_array, num_mutypes, dot_product, (3, 3)
        )
        denom_marg = denom.copy()
        denom_marg[-num_mutypes:] = 0
        expected = 2.0
        assert np.isclose(expected, result)

    def test_taylor_exponential(self):
        denom = np.arange(1, 5, dtype=int)
        var_array = np.array([1.0, 2.0, 5.0, 5.0], dtype=float) / 10
        diff_array = (2, 2)
        num_mutypes = 2
        time = 2
        dot_product = gfdiff.simple_dot_product_setback(denom, var_array, 2)
        result = gfdiff.taylor_coeff_exponential(
            -time,
            denom,
            dot_product,
            diff_array,
            num_mutypes,
            var_array[-1],
            (4, 4),
        )
        expected = 0.19322647367184684
        assert np.isclose(expected, result)

    def test_prod_of_polynomials(self, subsetdict):
        eq_matrix = np.array(
            [
                [[1, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 1, 0, 0], [1, 1, 1, 1]],
                [[1, 0, 0, 0], [1, 0, 1, 0]],
            ],
            dtype=np.int8,
        )
        mutype_shape = (2, 2)
        mutype_shape_with_marg = tuple(i + 1 for i in mutype_shape)
        size = np.prod(mutype_shape)
        mutype_array = np.array([i for i in np.ndindex(mutype_shape)], dtype=np.uint8)
        var_array = np.array([0.1, 0.2, 0.3, 0.3], dtype=np.float64)
        symbolic_var_array = np.array(
            [
                sympy.Rational(0.1),
                sympy.Rational(0.2),
                sympy.symbols("m1", real=True, positive=True),
                sympy.symbols("m2", real=True, positive=True),
            ],
            dtype=object,
        )
        result = gfdiff.compile_non_inverted_eq(
            eq_matrix, size, mutype_array, mutype_shape_with_marg
        )(var_array)
        combined_result = gfdiff.product_f(subsetdict, result).reshape(mutype_shape)
        # from symbolic eq
        subs_dict = {b: var_array[-1] for b in symbolic_var_array[-2:]}
        symbolic_eq = np.prod(
            gfdev.equations_from_matrix(eq_matrix, symbolic_var_array)
        )
        expected = return_symbolic_result(
            symbolic_eq, subs_dict, symbolic_var_array[-2:], mutype_shape
        )
        assert np.allclose(expected, combined_result)

    @pytest.mark.parametrize(
        "eq_matrix, delta_in_nom",
        [
            (
                np.array(
                    [
                        [[1, 0, 0, 0], [1, 1, 1, 1]],
                        [[0, 1, 0, 0], [2, 1, 0, 2]],
                        [[1, 0, 0, 0], [1, 1, 1, 0]],
                    ],
                    dtype=np.int8,
                ),
                True,
            ),
            (
                np.array(
                    [
                        [[1, 0, 0, 0], [1, 1, 1, 1]],
                        [[1, 0, 0, 0], [2, 1, 0, 2]],
                        [[1, 0, 0, 0], [1, 1, 1, 0]],
                    ],
                    dtype=np.int8,
                ),
                False,
            ),
        ],
    )
    def test_diff_inverse_laplace(self, eq_matrix, delta_in_nom, subsetdict):
        dummy_variable_idx = 1
        dummy_variable = sympy.symbols("E", positive=True, real=True)
        eq_matrix_no_delta = np.delete(eq_matrix, dummy_variable_idx, axis=2)
        mutype_shape = (2, 2)
        mutype_shape_with_marg = tuple(i + 1 for i in mutype_shape)
        mutype_array = np.array([i for i in np.ndindex(mutype_shape)], dtype=np.uint8)
        size = np.prod(mutype_shape)
        var_array = np.array([0.1, 0.3, 0.3], dtype=np.float64)
        symbolic_var_array = np.array(
            [
                sympy.Rational(0.1),
                dummy_variable,
                sympy.symbols("m1", positive=True, real=True),
                sympy.symbols("m2", positive=True, real=True),
            ],
            dtype=object,
        )
        time = 1.0
        result = gfdiff.compile_inverted_eq(
            eq_matrix_no_delta,
            size,
            subsetdict,
            delta_in_nom,
            mutype_array,
            mutype_shape_with_marg,
        )(var_array, time).reshape(mutype_shape)
        print("result", result)
        # from symbolic eq
        subs_dict = {b: var_array[-1] for b in symbolic_var_array[-2:]}
        T = sympy.symbols("T", positive=True, real=True)
        subs_dict[T] = time
        symbolic_eq = np.prod(
            gfdev.equations_from_matrix(eq_matrix, symbolic_var_array)
        )
        inverted_symbolic_eq = linverse.return_inverse_laplace_sympy(
            symbolic_eq, dummy_variable, T
        )
        expected = return_symbolic_result(
            inverted_symbolic_eq,
            subs_dict,
            symbolic_var_array[-2:],
            mutype_shape,
        )
        assert np.allclose(expected, result)

    @pytest.mark.parametrize(
        "eq_matrix, delta_in_denom, delta_in_nom",
        [
            (
                np.array(
                    [
                        [[1, 0, 0, 0], [1, 0, 1, 1]],
                        [[0, 1, 0, 0], [2, 1, 0, 2]],
                        [[1, 0, 0, 0], [1, 0, 1, 0]],
                    ],
                    dtype=np.int8,
                ),
                np.array([False, True, False], dtype=bool),
                True,
            ),
            (
                np.array(
                    [
                        [[1, 0, 0, 0], [1, 0, 1, 1]],
                        [[1, 0, 0, 0], [2, 1, 0, 2]],
                        [[1, 0, 0, 0], [1, 0, 1, 0]],
                    ],
                    dtype=np.int8,
                ),
                np.array([False, True, False], dtype=bool),
                False,
            ),
        ],
    )
    def test_diff_inverse_laplace2(
        self, eq_matrix, delta_in_denom, delta_in_nom, subsetdict
    ):
        # eq in which some factors have dummy variable, others don't
        dummy_variable_idx = 1
        dummy_variable = sympy.symbols("E", positive=True, real=True)
        eq_matrix_no_delta = np.delete(eq_matrix, dummy_variable_idx, axis=2)
        mutype_shape = (2, 2)
        mutype_shape_with_marg = (3, 3)
        mutype_array = np.array([i for i in np.ndindex((2, 2))], dtype=np.uint8)
        size = 4
        var_array = np.array([0.1, 0.3, 0.3], dtype=np.float64)
        symbolic_var_array = np.array(
            [
                sympy.Rational(0.1),
                dummy_variable,
                sympy.symbols("m1", positive=True, real=True),
                sympy.symbols("m2", positive=True, real=True),
            ],
            dtype=object,
        )
        time = 1.0
        result_inverted_part = gfdiff.compile_inverted_eq(
            eq_matrix_no_delta[delta_in_denom],
            size,
            subsetdict,
            delta_in_nom,
            mutype_array,
            mutype_shape_with_marg,
        )(var_array, time)
        result_non_inverted_part = gfdiff.compile_non_inverted_eq(
            eq_matrix_no_delta[~delta_in_denom],
            size,
            mutype_array,
            mutype_shape_with_marg,
        )(var_array)
        result = gfdiff.product_f(
            subsetdict,
            np.vstack((result_inverted_part[None, :], result_non_inverted_part)),
        ).reshape(mutype_shape)
        print("result", result)
        # from symbolic eq
        subs_dict = {b: var_array[-1] for b in symbolic_var_array[-2:]}
        T = sympy.symbols("T", positive=True, real=True)
        subs_dict[T] = time
        symbolic_eq = np.prod(
            gfdev.equations_from_matrix(eq_matrix, symbolic_var_array)
        )
        inverted_symbolic_eq = linverse.return_inverse_laplace_sympy(
            symbolic_eq, dummy_variable, T
        )
        expected = return_symbolic_result(
            inverted_symbolic_eq,
            subs_dict,
            symbolic_var_array[-2:],
            mutype_shape,
        )
        print("expected", expected)
        assert np.allclose(expected, result)

    @pytest.fixture(scope="class")
    def subsetdict(self):
        subsetdict = numba.typed.Dict()
        to_make_subsetdict = [
            np.array([0], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 2], dtype=np.uint8),
            np.array([0, 1, 2, 3], dtype=np.uint8),
        ]
        for idx, entry in enumerate(to_make_subsetdict):
            subsetdict[idx] = entry
        return subsetdict


def single_partial(ordered_mutype_list, partial):
    return list(
        itertools.chain.from_iterable(
            itertools.repeat(branchtype, count)
            for count, branchtype in zip(partial, ordered_mutype_list)
        )
    )


# how to substitute in sympy, how to differentiate
def return_symbolic_result(eq, subs_dict, branchtypes, shape, root=(0, 0)):
    symbolic_diff = np.zeros(shape, dtype=np.float64)
    for mutype in itertools.product(*(range(i) for i in shape)):
        if mutype == root:
            symbolic_diff[mutype] = eq.subs(subs_dict)
        else:
            how_to_diff = single_partial(branchtypes, mutype)
            symbolic_diff[mutype] = (
                1
                / np.prod(factorial(mutype))
                * sympy.diff(eq, *how_to_diff).subs(subs_dict)
            )

    return symbolic_diff

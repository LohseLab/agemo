import itertools
import numpy as np
import numba
import pytest
import sage.all
import scipy.stats as stats

from scipy.special import factorial

import agemo.gflib as gflib
import agemo.diff as gfdiff
import agemo.mutations as gfmuts
import agemo.evaluate as gfeval
import agemo.events as eventslib
import agemo.legacy.inverse as linverse
import agemo.legacy.mutations as smuts
import tests.gfdev as gfdev


class TestDotProduct:
    def test_casc_dot_product(self):
        rng = np.random.default_rng()
        A = rng.random(20)
        B = rng.random(20)
        assert np, isclose(
            gfdiff.casc_dot_product(A, B), gfdiff.simple_dot_product(A, B)
        )


@pytest.mark.diff
@pytest.mark.taylor
class Test_taylor:
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
                sage.all.Rational(0.1),
                sage.all.Rational(0.2),
                sage.all.SR.var("m1"),
                sage.all.SR.var("m2"),
            ],
            dtype=object,
        )
        result = gfdiff.compile_non_inverted_eq(
            eq_matrix, size, mutype_array, mutype_shape_with_marg
        )(var_array)
        # subsetdict = {0: np.array([0], dtype=np.uint8),
        # 			1: np.array([0,1], dtype=np.uint8),
        # 			2: np.array([0,2], dtype=np.uint8),
        # 			3: np.array([0,1,2,3], dtype=np.uint8)}
        combined_result = gfdiff.product_f(subsetdict, result).reshape(mutype_shape)
        # from symbolic eq
        subs_dict = {b: var_array[-1] for b in symbolic_var_array[-2:]}
        symbolic_eq = np.prod(
            gflib.equations_from_matrix(eq_matrix, symbolic_var_array)
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
        dummy_variable = sage.all.SR.var("E")
        eq_matrix_no_delta = np.delete(eq_matrix, dummy_variable_idx, axis=2)
        mutype_shape = (2, 2)
        mutype_shape_with_marg = tuple(i + 1 for i in mutype_shape)
        mutype_array = np.array([i for i in np.ndindex(mutype_shape)], dtype=np.uint8)
        size = np.prod(mutype_shape)
        var_array = np.array([0.1, 0.3, 0.3], dtype=np.float64)
        symbolic_var_array = np.array(
            [
                sage.all.Rational(0.1),
                dummy_variable,
                sage.all.SR.var("m1"),
                sage.all.SR.var("m2"),
            ],
            dtype=object,
        )
        time = 1.0
        # subsetdict = numba.typed.Dict()
        # to_make_dict = [
        # 			np.array([0], dtype=np.uint8),
        # 			np.array([0,1], dtype=np.uint8),
        # 			np.array([0,2], dtype=np.uint8),
        # 			np.array([0,1,2,3], dtype=np.uint8)
        # 			]
        # for idx, entry in enumerate(to_make_subsetdict):
        # 	subsetdict[idx] = entry
        result = gfdiff.compile_inverted_eq(
            eq_matrix_no_delta,
            size,
            subsetdict,
            delta_in_nom,
            mutype_array,
            mutype_shape_with_marg,
        )(var_array, time).reshape(mutype_shape)
        print(result)
        # from symbolic eq
        subs_dict = {b: var_array[-1] for b in symbolic_var_array[-2:]}
        subs_dict[sage.all.SR.var("T")] = time
        symbolic_eq = np.prod(
            gflib.equations_from_matrix(eq_matrix, symbolic_var_array)
        )
        inverted_symbolic_eq = linverse.return_inverse_laplace(
            symbolic_eq, dummy_variable
        )
        expected = return_symbolic_result(
            inverted_symbolic_eq,
            subs_dict,
            symbolic_var_array[-2:],
            mutype_shape,
        )
        print(expected)
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
        dummy_variable = sage.all.SR.var("E")
        eq_matrix_no_delta = np.delete(eq_matrix, dummy_variable_idx, axis=2)
        mutype_shape = (2, 2)
        mutype_shape_with_marg = (3, 3)
        mutype_array = np.array([i for i in np.ndindex((2, 2))], dtype=np.uint8)
        size = 4
        var_array = np.array([0.1, 0.3, 0.3], dtype=np.float64)
        symbolic_var_array = np.array(
            [
                sage.all.Rational(0.1),
                dummy_variable,
                sage.all.SR.var("m1"),
                sage.all.SR.var("m2"),
            ],
            dtype=object,
        )
        time = 1.0
        # subsetdict = numba.typed.Dict()
        # to_make_dict = [
        # 			np.array([0], dtype=np.uint8),
        # 			np.array([0,1], dtype=np.uint8),
        # 			np.array([0,2], dtype=np.uint8),
        # 			np.array([0,1,2,3], dtype=np.uint8)
        # 			]
        # for idx, entry in to_make_dict:
        # 	subsetdict[idx] = entry
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
        print("result")
        print(result)
        # from symbolic eq
        subs_dict = {b: var_array[-1] for b in symbolic_var_array[-2:]}
        subs_dict[sage.all.SR.var("T")] = time
        symbolic_eq = np.prod(
            gflib.equations_from_matrix(eq_matrix, symbolic_var_array)
        )
        inverted_symbolic_eq = linverse.return_inverse_laplace(
            symbolic_eq, dummy_variable
        )
        expected = return_symbolic_result(
            inverted_symbolic_eq,
            subs_dict,
            symbolic_var_array[-2:],
            mutype_shape,
        )
        print("expected")
        print(expected)
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


def return_symbolic_result(eq, subs_dict, branchtypes, shape, root=(0, 0)):
    symbolic_diff = np.zeros(shape, dtype=np.float64)
    for mutype in itertools.product(*(range(i) for i in shape)):
        if mutype == root:
            symbolic_diff[mutype] = eq.subs(subs_dict)
        else:
            how_to_diff = smuts.single_partial(branchtypes, mutype)
            symbolic_diff[mutype] = (
                1
                / np.prod(factorial(mutype))
                * sage.all.diff(eq, *how_to_diff).subs(subs_dict)
            )

    return symbolic_diff


@pytest.mark.diff
@pytest.mark.collapse
class Test_collapse_graph:
    def test_collapse_graph(self):
        graph_array = ((1, 2, 4, 5), (2,), (3,), (6,), (3,), (3,), tuple())
        eq_matrix = np.array(
            [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 0, 0]]],
            dtype=np.uint8,
        )
        adjacency_matrix = np.full(
            (len(graph_array), len(graph_array)), fill_value=255, dtype=np.int8
        )
        adjacency_matrix[np.array([0, 0, 1]), np.array([1, 2, 2])] = 0
        adjacency_matrix[np.array([2]), np.array([3])] = 1
        adjacency_matrix[np.array([0, 0, 3, 4, 5]), np.array([4, 5, 6, 3, 3])] = 2
        sample_list = [("a", "a"), ("b", "b")]
        btc = gfmuts.BranchTypeCounter(sample_list)
        pse = eventslib.PopulationSplitEvent(2, 0, 1)
        gfObj = gflib.GfMatrixObject(
            btc,
            [
                pse,
            ],
        )

        (
            collapsed_graph_array,
            adjacency_matrix_b,
            eq_array,
            to_invert_array,
        ) = gfObj.collapse_graph(graph_array, adjacency_matrix, eq_matrix)
        expected_graph_array = ((1, 2, 5, 6), (3,), (3,), (4,), (), (4,), (4,))
        expected_to_invert_array = np.zeros(9, dtype=bool)
        expected_to_invert_array[-2:] = 1
        expected_eq_array = (
            (2,),
            (2,),
            (2,),
            (2,),
            (2,),
            (2,),
            (2,),
            (0, 1),
            (0, 0, 1),
        )

        def compare(ar1, ar2):
            for a, b in zip(ar1, ar2):
                assert np.array_equal(np.array(a), np.array(b))

        compare(expected_graph_array, collapsed_graph_array)
        compare(expected_eq_array, eq_array)
        assert np.array_equal(expected_to_invert_array, to_invert_array)

    @pytest.mark.parametrize(
        "sample_list, exp_graph_array",
        [
            ([(), ("a", "a")], [(1, 3), (2,), (), ()]),
            (
                [(), ("a", "a", "a")],
                [(1, 4, 5), (2,), (3,), (), (3,), ()],
            ),
        ],
    )
    def test_graph_with_multiple_endpoints(self, sample_list, exp_graph_array):
        gfobj = self.get_gf_no_mutations(sample_list)
        graph_array, adjacency_matrix, eq_matrix = gfobj.make_graph()
        collapsed_graph_array, *_ = gfobj.collapse_graph(
            graph_array, adjacency_matrix, eq_matrix
        )
        print(exp_graph_array)
        print(collapsed_graph_array)
        for o, e in zip(collapsed_graph_array, exp_graph_array):
            assert o == e

    def get_gf_no_mutations(self, sample_list):
        pse = eventslib.PopulationSplitEvent(2, 0, 1)
        btc = gfmuts.BranchTypeCounter(sample_list, rooted=True)
        gfobj = gflib.GfMatrixObject(
            btc,
            [
                pse,
            ],
        )
        return gfobj


@pytest.mark.diff
@pytest.mark.taylor_single
class Test_taylor_single_pop:
    @pytest.mark.parametrize("size", [2, 3])
    def test_combining_probabilities(self, size):
        sample_list = [(), ("a",) * size]
        btc = self.get_BT_object(sample_list)
        gfobj = self.get_gf_no_mutations(btc)
        mtc = self.get_MT_object(size - 1, btc)

        max_k = np.full(size - 1, fill_value=2, dtype=int)
        shape = tuple(max_k + 2)
        variable_array = np.array([1.0, 2.0], dtype=np.float64)
        theta = 0.5
        theta_array = np.full(size - 1, fill_value=theta)
        variable_array = np.hstack((variable_array, theta_array))
        time = 1.5

        gfEvalObj = gfeval.BSFSEvaluator(gfobj, mtc)
        result_with_marginals = evaluate_no_marginal(
            gfEvalObj, theta, variable_array, time
        )
        result_with_marginals = result_with_marginals.reshape(shape)

        # symbolic derivation
        ordered_mutype_list = [sage.all.SR.var(f"m_{idx}") for idx in range(1, size)]
        # num_mutypes = len(ordered_mutype_list)
        alt_variable_array = np.hstack(
            (variable_array[:2], np.array(ordered_mutype_list))
        )
        # mutype_array = np.array([t for t in np.ndindex(shape)], dtype=np.uint8)

        print(result_with_marginals)
        exp_result = evaluate_symbolic_equation(
            gfobj,
            ordered_mutype_list,
            max_k,
            theta,
            alt_variable_array,
            time,
            gfobj.discrete_events[0],
        )
        print(exp_result)
        assert np.allclose(exp_result, result_with_marginals)

    def get_BT_object(self, sample_list):
        return gfmuts.BranchTypeCounter(sample_list, rooted=True)

    def get_MT_object(self, size, btc):
        shape = (4,) * size
        return gfmuts.MutationTypeCounter(btc, shape)

    def get_gf_no_mutations(self, btc):
        pse = eventslib.PopulationSplitEvent(2, 0, 1)
        gfobj = gflib.GfMatrixObject(
            btc,
            [
                pse,
            ],
        )
        return gfobj


@pytest.mark.diff
@pytest.mark.taylor_IM
class Test_taylorIM:
    def test_IM_models(self, get_IM_gfobject, get_MT_object):
        gfobj, parameter_combo, model, BranchTypeCounter = get_IM_gfobject
        max_k = np.array([2, 2, 2, 2], dtype=int)
        # shape = tuple(max_k + 2)
        # variables depending on model: c0, c1, c2, M, E
        theta, variable_array, time = parameter_combo
        theta_array = np.full(len(max_k), fill_value=theta)
        var = np.hstack((variable_array, theta_array))

        gfEvalObj = gfeval.BSFSEvaluator(gfobj, get_MT_object)
        result = gfEvalObj.evaluate(theta, var, time)
        self.compare_ETPs_model(model, result)

    def compare_ETPs_model(self, model, ETPs):
        precalc_ETPs = np.squeeze(np.load(f"tests/ETPs/{model}.npy"))
        print(precalc_ETPs)
        print("--------------")
        print(ETPs)
        assert np.allclose(precalc_ETPs, ETPs)

    @pytest.fixture(scope="class")
    def get_BT_object(self):
        sample_list = [(), ("a", "a"), ("b", "b")]
        branchtype_dict_mat = gfdev.make_branchtype_dict_idxs_gimble()
        return gfmuts.BranchTypeCounter(
            sample_list, branchtype_dict=branchtype_dict_mat
        )

    @pytest.fixture(
        scope="class",
        params=[
            (
                (1, 2, 0),
                True,
                None,
                None,
                [
                    72 / 125,
                    np.array([1.0, 15 / 13, 5 / 2], dtype=np.float64),
                    10 / 3,
                ],
                "DIV",
            ),
            (
                None,
                None,
                (2, 1),
                True,
                [
                    312 / 625,
                    np.array(
                        [0.0, 1.0, 13 / 6, 134369693800271 / 73829502088061],
                        dtype=np.float64,
                    ),
                    0.0,
                ],
                "MIG_BA",
            ),
            (
                (1, 2, 0),
                True,
                (1, 2),
                True,
                [
                    72 / 125,
                    np.array([1.0, 15 / 13, 5 / 2, 21 / 10], dtype=np.float64),
                    10 / 3,
                ],
                "IM_AB",
            ),
        ],
        ids=["DIV", "MIG", "IM"],
    )
    def get_IM_gfobject(self, request, get_BT_object):
        return get_IM_gfobject_BT(request.param, get_BT_object)

    @pytest.fixture(scope="class")
    def get_MT_object(self, get_BT_object):
        shape = (4, 4, 4, 4)
        return gfmuts.MutationTypeCounter(get_BT_object, shape)


@pytest.mark.diff
@pytest.mark.epsilon
class Test_epsilon:
    def test_IM_models(self, get_IM_gfobject, get_MT_object):
        gfobj, parameter_combo, model, BranchTypeCounter = get_IM_gfobject
        max_k = np.array([2, 2, 2, 2], dtype=int)
        # shape = tuple(max_k + 2)
        # variables depending on model: c0, c1, c2, M, E
        theta, variable_array, time = parameter_combo
        theta_array = np.full(len(max_k), fill_value=theta)
        var = np.hstack((variable_array, theta_array))

        gfEvalObj = gfeval.BSFSEvaluator(gfobj, get_MT_object)
        result_with_marginals = gfEvalObj.evaluate(theta, var, time)
        sim_counts = np.load("tests/ETPs/IM_BA_epsilon_sim.npy")
        result_with_marginals[result_with_marginals < 0] = 0
        expected_counts = result_with_marginals * np.sum(sim_counts)
        chisquare(sim_counts, expected_counts)

    def test_IM_to_DIV_simplfication(self, get_IM_gfobject, get_MT_object):
        gfobj, _, model, BranchTypeCounter = get_IM_gfobject
        max_k = np.array([2, 2, 2, 2], dtype=int)
        # shape = tuple(max_k + 2)
        # variables depending on model: c0, c1, c2, M, E

        theta = 0.5169514294907595
        time = 0.17951396341318912
        var = np.array(
            [
                1,
                1,
                4.211515409328338,
                0,
                0.5169514294907595,
                0.5169514294907595,
                0.5169514294907595,
                0.5169514294907595,
            ]
        )

        for seed in [152, 245555, 1224556, 1, 42]:
            gfEvalObj = gfeval.BSFSEvaluator(gfobj, get_MT_object, seed)
            result_with_marginals = gfEvalObj.evaluate(theta, var, time)
            assert np.isclose(np.sum(result_with_marginals), 1.0)
            assert np, all(result_with_marginals >= 0)
            assert np, all(result_with_marginals < 1)

    @pytest.fixture(scope="class")
    def get_BT_object(self):
        sample_list = [(), ("a", "a"), ("b", "b")]
        branchtype_dict_mat = gfdev.make_branchtype_dict_idxs_gimble()
        return gfmuts.BranchTypeCounter(
            sample_list, branchtype_dict=branchtype_dict_mat
        )

    @pytest.fixture(
        scope="class",
        params=[
            (
                (1, 2, 0),
                True,
                (2, 1),
                True,
                [0.5, np.array([1.0, 0.2, 0.3, 0.4], dtype=np.float64), 1.5],
                "IM_BA_zerodivision",
            ),
        ],
    )
    def get_IM_gfobject(self, request, get_BT_object):
        return get_IM_gfobject_BT(request.param, get_BT_object)

    @pytest.fixture(scope="class")
    def get_MT_object(self, get_BT_object):
        shape = (4, 4, 4, 4)
        return gfmuts.MutationTypeCounter(get_BT_object, shape)


def evaluate_symbolic_equation(
    gfobj, ordered_mutype_list, max_k, theta, var, time, delta_idx, sage_inverse=False
):
    theta = sage.all.Rational(theta)
    rate_dict = {b: theta for b in ordered_mutype_list}
    paths, eq_matrix = gfobj.make_gf()
    if not sage_inverse:
        alt_eqs = gfdev.equations_from_matrix_with_inverse(
            eq_matrix, paths, var, time, delta_idx
        )
    else:
        alt_eqs = gfdev.equations_with_sage(
            eq_matrix, paths, var, sage.all.Rational(time), delta_idx
        )
    gf_alt = sum(alt_eqs)
    result = smuts.depth_first_mutypes(
        max_k, ordered_mutype_list, gf_alt, theta, rate_dict
    )
    return result.astype(np.float64)


def get_IM_gfobject_BT(params, btc):
    # ancestral_pop = 0
    coalescence_rate_idxs = (0, 1, 2)
    num_variables = len(coalescence_rate_idxs)
    (
        exodus_direction,
        exodus_rate,
        migration_direction,
        migration_rate,
        variable_array,
        model,
    ) = params

    events_list = []
    migration_rate_idx, exodus_rate_idx = None, None
    if migration_rate is not None:
        mige = eventslib.MigrationEvent(num_variables, *migration_direction)
        num_variables += 1
        events_list.append(mige)
    if exodus_rate is not None:
        *derived, ancestral = exodus_direction
        pse = eventslib.PopulationSplitEvent(num_variables, ancestral, *derived)
        num_variables += 1
        events_list.append(pse)

    # gfobj = gflib.GfMatrixObject(
    #    btc.sample_configuration,
    #    coalescence_rate_idxs,
    #    btc.labels_dict,
    #    exodus_rate=exodus_rate_idx,
    #    exodus_direction=exodus_direction,
    #    migration_rate=migration_rate_idx,
    #    migration_direction=migration_direction,
    # )
    gfobj = gflib.GfMatrixObject(btc, events_list)

    return gfobj, variable_array, model, btc


def chisquare(observed, expected, p_th=0.05, recombination=False, all_sims=False):
    # expected counts need to be larger than 5 for chisquare
    # combining bins with less than 5 counts into one bin.
    obs = np.reshape(observed, -1)
    exp = np.reshape(expected, -1)
    if not (recombination or all_sims):
        assert np.all(
            obs[exp == 0] == 0
        ), "chisquare error: at least one cell with expected frequency 0 has observed values"
    # bin all values with counts smaller than 5
    binning_idxs = np.digitize(exp, np.array([5]), right=True)
    # exp_smaller = np.sum(exp[binning_idxs == 0])
    # obs_smaller = np.sum(obs[binning_idxs == 0])
    exp_binned = exp[binning_idxs == 1]
    obs_binned = obs[binning_idxs == 1]
    # make sure exp_binned and obs_binned sum to same value
    exp_binned *= np.sum(obs_binned) / np.sum(exp_binned)
    not_zeros = exp_binned > 0
    if sum(not_zeros) < 1:
        assert False  # expected probabilities are all 0
    else:
        chisq = stats.chisquare(obs_binned[not_zeros], exp_binned[not_zeros])
        print("chisquare value:", chisq)
        print("exp", exp_binned)
        print("obs", obs_binned)
        assert chisq.pvalue > p_th


def evaluate_no_marginal(gfEvaluatorObj, theta, var, time):
    results = gfEvaluatorObj.evaluator(var, time)
    final_result = gfdiff.iterate_eq_graph(
        gfEvaluatorObj.dependency_sequence,
        gfEvaluatorObj.eq_graph_array,
        results,
        gfEvaluatorObj.subsetdict,
    )
    theta_multiplier_matrix = gfdiff.taylor_to_probability(
        gfEvaluatorObj.multiplier_matrix, theta
    )
    return theta_multiplier_matrix * final_result

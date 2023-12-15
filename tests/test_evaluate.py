import itertools
import numpy as np
import numba
import pytest
import scipy.stats as stats
import sympy

import agemo.gflib as gflib
import agemo.diff as gfdiff
import agemo.mutations as gfmuts
import agemo.evaluate as gfeval
import agemo.events as eventslib

import agemo.legacy.mutations as smuts

import tests.gfdev as gfdev

@pytest.mark.diff
@pytest.mark.taylor_single
class TestTaylorSinglePop:
    @pytest.mark.parametrize("size", [2, 3])
    def test_combining_probabilities(self, size):
        sample_list = [(), ("a",) * size]
        btc = self.get_BT_object(sample_list)
        # there should not be any discrete events here!!!
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
        print('obs', result_with_marginals)
        
        # symbolic derivation
        ordered_mutype_list = [sympy.symbols(f"m_{idx}", positive=True, real=True) for idx in range(1, size)]
        # num_mutypes = len(ordered_mutype_list)
        symbolic_variable_array = np.hstack(
            (variable_array[:2], np.array(ordered_mutype_list))
        )
        print('discrete', gfobj.discrete_events[0])
        exp_result = evaluate_symbolic_equation(
            gfobj,
            ordered_mutype_list,
            max_k,
            theta,
            symbolic_variable_array,
            time,
            gfobj.discrete_events[0],
            sage_inverse=True
        )
        print('exp_result', exp_result)
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
class TestTaylorIM:
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
class TestEpsilon:
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
    theta = sympy.Rational(theta)
    rate_dict = {b: theta for b in ordered_mutype_list}
    paths, eq_matrix = gfobj.make_gf()
    if not sage_inverse:
        alt_eqs = gfdev.equations_from_matrix_with_inverse(
            eq_matrix, paths, var, time, delta_idx
        )
    else:
        delta = sympy.symbols('d', positive=True, real=True) if delta_idx is not None else None
        var_array = np.insert(var, delta_idx, delta)
        alt_eqs = gfdev.equations_with_sympy(
            eq_matrix, paths, var_array, sympy.Rational(time), delta
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
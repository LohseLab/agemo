import numpy as np
import pytest
import sage.all

import tests.gfdev as gfdev
import agemo.gflib as gflib
import agemo.mutations as mut
import agemo.legacy.mutations as smut
import agemo.legacy.evaluate as leval

all_configs = {
    "IM_AB": {
        "global_info": {
            "model_file": "models/IM_AB.tsv",
            "mu": 3e-9,
            "ploidy": 2,
            "sample_pop_ids": ["A", "B"],
            "blocklength": 64,
            "k_max": {"m_1": 2, "m_2": 2, "m_3": 2, "m_4": 2},
            "reference_pop": "A_B",
        },
        "sim_configs": [
            {
                "Ne_A": 1.3e6,
                "Ne_B": 6e5,
                "Ne_A_B": 1.5e6,
                "T": 1e7,
                "me_A_B": 7e-7,
                "recombination": 0,
            }
        ],
        "gf_vars": {
            "sample_list": [(), ("a", "a"), ("b", "b")],
            "migration_rate": sage.all.SR.var("M"),
            "migration_direction": [(1, 2)],
            "exodus_rate": sage.all.var("E"),
            "exodus_direction": [(1, 2, 0)],
            "ancestral_pop": 0,
        },
    },
    "DIV": {
        "global_info": {
            "model_file": "models/DIV.tsv",
            "mu": 3e-9,
            "ploidy": 2,
            "sample_pop_ids": ["A", "B"],
            "blocklength": 64,
            "k_max": {"m_1": 2, "m_2": 2, "m_3": 2, "m_4": 2},
            "reference_pop": "A_B",
        },
        "sim_configs": [
            {
                "Ne_A": 1.3e6,
                "Ne_B": 6e5,
                "Ne_A_B": 1.5e6,
                "T": 1e7,
                "recombination": 0,
            }
        ],
        "gf_vars": {
            "sample_list": [(), ("a", "a"), ("b", "b")],
            "exodus_rate": sage.all.SR.var("E"),
            "exodus_direction": [(1, 2, 0)],
            "ancestral_pop": 0,
        },
    },
    "MIG_BA": {
        "global_info": {
            "model_file": "models/MIG_BA.tsv",
            "mu": 3e-9,
            "ploidy": 2,
            "sample_pop_ids": ["A", "B"],
            "blocklength": 64,
            "k_max": {"m_1": 2, "m_2": 2, "m_3": 2, "m_4": 2},
            "reference_pop": "A",
        },
        "sim_configs": [
            {"Ne_A": 1.3e6, "Ne_B": 6e5, "me_B_A": 7e-7, "recombination": 0}
        ],
        "gf_vars": {
            "sample_list": [(), ("a", "a"), ("b", "b")],
            "migration_rate": sage.all.SR.var("M"),
            "migration_direction": [(2, 1)],
            "ancestral_pop": 1,
        },
    },
}


@pytest.mark.simple_models
class Test_gf_simple:
    def test_generate_ETPs(self):
        sample_list = [("a", "b")]
        branchtype_dict = smut.make_branchtype_dict(
            sample_list, mapping="label"
        )
        theta = 0.5  # this is actually theta/2
        ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
        rate_dict = {branchtype: theta for branchtype in ordered_mutype_list}
        coalescence_rates = (1,)
        gfobj = gflib.GFObject(sample_list, coalescence_rates, branchtype_dict)
        gf = sum(gfobj.make_gf())
        max_k = np.array([2, 2])
        all_mutation_configurations = list(mut.return_mutype_configs(max_k))
        root = tuple(0 for _ in max_k)
        mutype_tree = mut.make_mutype_tree(
            all_mutation_configurations, root, max_k
        )
        result = smut.make_result_dict_from_mutype_tree_alt(
            gf, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k
        )
        result = result.astype(np.float64)
        check = np.array(
            [
                [0.5, 0.125, 0.03125, 0.66666667],
                [0.125, 0.0625, 0.0234375, 0.22222222],
                [0.03125, 0.0234375, 0.01171875, 0.07407407],
                [0.66666667, 0.22222222, 0.07407407, 1],
            ]
        )
        assert np.allclose(result, check)

    def test_probK(self):
        # test probability of seeing 1 mutation on each branch
        ordered_mutype_list = [sage.all.SR.var("z_a"), sage.all.SR.var("z_b")]
        theta = sage.all.Rational(0.5)
        gf = 1 / (sum(ordered_mutype_list) + 1)
        # partials = ordered_mutype_list[:]
        marginals = None
        ratedict = {mutype: theta for mutype in ordered_mutype_list}
        mucounts = np.array([1, 1])
        # mucount_total = 2
        # mucount_fact_prod = 1
        probk_deriv = smut.generate_equation(
            gf, (0, 0), (1, 1), (2, 2), ordered_mutype_list, marginals
        )
        probk = smut.eval_equation(probk_deriv, theta, ratedict, mucounts, 165)
        assert float(probk) == float(
            1 / 2 * (2 * theta) ** 2 / (2 * theta + 1) ** 3
        )


@pytest.mark.zero_division
class Test_zero_division:
    @pytest.fixture(scope="class")
    def get_gf(self):
        config = {
            "sample_list": [(), ("a", "a"), ("b", "b")],
            "coalescence_rates": (
                sage.all.SR.var("c0"),
                sage.all.SR.var("c1"),
                sage.all.SR.var("c2"),
            ),
            "k_max": {"m_1": 2, "m_2": 2, "m_3": 2, "m_4": 2},
            "migration_rate": sage.all.SR.var("M"),
            "migration_direction": [(1, 2)],
            "exodus_rate": sage.all.SR.var("E"),
            "exodus_direction": [(1, 2, 0)],
        }
        mutype_labels, max_k = zip(*sorted(config["k_max"].items()))
        config["mutype_labels"] = mutype_labels
        del config["k_max"]
        gf = leval.get_gf(**config)
        return (gf, mutype_labels, max_k)

    def test_zero_div(self, get_gf):
        gf, mutype_labels, max_k = get_gf
        coalescence_rates = (
            sage.all.SR.var("c0"),
            sage.all.SR.var("c1"),
            sage.all.SR.var("c2"),
        )
        coal_rates_values = {
            c: sage.all.Rational(v)
            for c, v in zip(coalescence_rates, (3, 1, 3))
        }
        parameter_dict = {
            sage.all.SR.var("M"): sage.all.Rational(2),
            sage.all.SR.var("T"): sage.all.Rational(1),
        }
        parameter_dict = {**parameter_dict, **coal_rates_values}
        # theta = sage.all.Rational(1)
        epsilon = sage.all.Rational(0.00001)
        gfEvaluatorObj = leval.gfEvaluator(
            gf,
            max_k,
            mutype_labels,
            exclude=[
                (2, 3),
            ],
        )
        result = gfEvaluatorObj._subs_params(parameter_dict, epsilon)
        assert isinstance(result, sage.symbolic.expression.Expression)

    def test_IM_to_DIV(self, get_gf):
        gf, mutype_labels, max_k = get_gf
        coalescence_rates = (
            sage.all.SR.var("c0"),
            sage.all.SR.var("c1"),
            sage.all.SR.var("c2"),
        )
        coal_rates_values = {
            c: sage.all.Rational(v)
            for c, v in zip(coalescence_rates, (1, 1, 1))
        }
        parameter_dict = {
            sage.all.SR.var("M"): sage.all.Rational(0),
            sage.all.SR.var("T"): sage.all.Rational(1),
        }
        parameter_dict = {**parameter_dict, **coal_rates_values}
        # theta = sage.all.Rational(1)
        epsilon = sage.all.Rational(0.00001)
        gfEvaluatorObj = leval.gfEvaluator(
            gf,
            max_k,
            mutype_labels,
            exclude=[
                (2, 3),
            ],
        )
        result = gfEvaluatorObj._subs_params(parameter_dict, epsilon)
        assert isinstance(result, sage.symbolic.expression.Expression)


@pytest.mark.gimble
class Test_to_gimble:
    @pytest.mark.parametrize("model", [("IM_AB"), ("DIV"), ("MIG_BA")])
    def test_ETPs(self, model):
        ETPs = self.calculate_ETPs(model)
        self.compare_ETPs_model(model, ETPs)

    def calculate_ETPs(self, model):
        config = all_configs[model]
        config["sim_config"] = config["sim_configs"][0]
        del config["sim_configs"]
        coalescence_rates = tuple(
            sage.all.SR.var(c) for c in ["c0", "c1", "c2"]
        )
        migration_rate = config["gf_vars"].get("migration_rate")
        migration_direction = config["gf_vars"].get("migration_direction")
        exodus_rate = config["gf_vars"].get("exodus_rate")
        exodus_direction = config["gf_vars"].get("exodus_direction")
        mutype_labels, max_k = zip(
            *sorted(config["global_info"]["k_max"].items())
        )
        gf = leval.get_gf(
            config["gf_vars"]["sample_list"],
            coalescence_rates,
            mutype_labels,
            migration_direction=migration_direction,
            migration_rate=migration_rate,
            exodus_direction=exodus_direction,
            exodus_rate=exodus_rate,
        )
        gfEvaluatorObj = leval.gfEvaluator(
            gf,
            max_k,
            mutype_labels,
            exclude=[
                (2, 3),
            ],
        )
        parameter_dict = gfdev.get_parameter_dict(coalescence_rates, **config)
        theta = gfdev.get_theta(**config)
        return gfEvaluatorObj.evaluate_gf(parameter_dict, theta)

    def compare_ETPs_model(self, model, ETPs):
        gimbled_ETPs = np.squeeze(np.load(f"tests/ETPs/{model}.npy"))
        assert np.all(np.isclose(gimbled_ETPs, ETPs))

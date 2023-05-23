import numpy as np
import pytest

import agemo.mutations as mut
import agemo.legacy.evaluate as leval
import agemo.legacy.mutations as smut


@pytest.mark.muts
class Test_mutypetree:
    def test_incomplete_tree(self):
        all_mutypes = sorted(
            [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (2, 1, 0, 0),
                (2, 1, 2, 0),
                (0, 0, 2, 2),
                (0, 0, 3, 2),
                (2, 1, 3, 0),
                (2, 1, 2, 3),
            ]
        )
        root = (0, 0, 0, 0)
        max_k = (2, 2, 2, 2)
        mutype_tree = mut.make_mutype_tree(all_mutypes, root, max_k)
        result = {
            (0, 0, 0, 0): [(0, 0, 2, 2), (1, 0, 0, 0)],
            (0, 0, 0, 2): [
                (0, 0, 3, 2),
            ],
            (1, 0, 0, 0): [
                (2, 1, 0, 0),
            ],
            (2, 1, 0, 0): [(2, 1, 2, 0), (2, 1, 3, 0)],
            (2, 1, 2, 0): [(2, 1, 2, 3)],
        }
        assert result == mutype_tree

    def test_complete_tree(self):
        root = (0, 0, 0, 0)
        max_k = (2, 2, 2, 2)
        all_mutypes = sorted(mut.return_mutype_configs(max_k))
        mutype_tree = mut.make_mutype_tree(all_mutypes, root, max_k)
        mutype_tree_single_digit = mut.make_mutype_tree_single_digit(
            all_mutypes, root, max_k
        )
        assert mutype_tree == mutype_tree_single_digit

    def test_muts_to_gimble(self):
        gf = 1
        max_k = (2, 2, 2, 2)
        # root = (0, 0, 0, 0)
        exclude = [
            (2, 3),
        ]
        restrict_to = sorted(
            [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (2, 1, 0, 0),
                (2, 1, 2, 0),
                (0, 0, 2, 2),
                (0, 0, 3, 2),
                (2, 1, 3, 0),
                (2, 1, 2, 3),
            ]
        )
        mutypes = tuple(f"m_{idx}" for idx in range(len(max_k)))
        gfEvaluatorObj = leval.GfEvaluator(
            gf, max_k, mutypes, exclude=exclude, restrict_to=restrict_to
        )

        print(gfEvaluatorObj.mutype_tree)
        # types to add:
        # (0,0,0,2),(0,0,1,2),(0,0,2,2),(2,1,2,0),(2,1,2,1),(2,1,2,2),(2,1,0,0),(2,1,1,0),(2,1,2,0)
        expected_result = {
            (0, 0, 0, 0): [(0, 0, 0, 2), (1, 0, 0, 0)],
            (1, 0, 0, 0): [
                (2, 1, 0, 0),
            ],
            (2, 1, 0, 0): [(2, 1, 1, 0), (2, 1, 3, 0)],
            (2, 1, 1, 0): [
                (2, 1, 2, 0),
            ],
        }
        assert gfEvaluatorObj.mutype_tree == expected_result

    def test_list_marginal_idxs(self):
        marginals = np.array([(0, 0, 3, 1), (3, 0, 1, 3)], dtype=np.uint8)
        max_k = (2, 2, 2, 2)
        result = [smut.list_marginal_idxs(m, max_k) for m in marginals]
        expected_result = [
            [(0, 0, 0, 1), (0, 0, 1, 1), (0, 0, 2, 1)],
            [
                (0, 0, 1, 0),
                (0, 0, 1, 1),
                (0, 0, 1, 2),
                (1, 0, 1, 0),
                (1, 0, 1, 1),
                (1, 0, 1, 2),
                (2, 0, 1, 0),
                (2, 0, 1, 1),
                (2, 0, 1, 2),
            ],
        ]
        assert result == expected_result

    def test_add_marginals_restrict_to(self):
        restrict_to = [
            (0, 0, 3, 1),
            (3, 0, 1, 3),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (1, 0, 0, 0),
        ]
        max_k = (2, 2, 2, 2)
        result = smut.add_marginals_restrict_to(restrict_to, max_k)
        expected_result = sorted(
            [
                (0, 0, 0, 1),
                (0, 0, 1, 1),
                (0, 0, 2, 1),
                (0, 0, 3, 1),
                (0, 0, 1, 0),
                (0, 0, 1, 2),
                (1, 0, 1, 0),
                (1, 0, 1, 1),
                (1, 0, 1, 2),
                (2, 0, 1, 0),
                (2, 0, 1, 1),
                (2, 0, 1, 2),
                (3, 0, 1, 3),
                (1, 0, 0, 0),
            ]
        )
        assert expected_result == result


class TestTypeCounter:
    def test_type_counter(self):
        tc = mut.BranchTypeCounter(
            [("a", "a"), ("b", "b")],
        )
        assert tc.phased == False
        assert tc.rooted == False
        assert tc.labels == ["a", "b", "aa", "ab"]
        assert tc.samples_per_pop == (2, 2)
        assert tc.compatibility_check == [set(), set(), {3}, {2}]

    def test_mutype_counter(self):
        tc = mut.BranchTypeCounter(
            [("a", "b"), ("c", "d")],
            phased=True,
        )
        mutype_shape = tuple([2 for _ in range(len(tc))])
        with pytest.raises(ValueError):
            mc = mut.MutationTypeCounter(tc, mutype_shape)

    def test_mutype_counter2(self):
        tc = mut.BranchTypeCounter(
            [("a", "a"), ("b", "b")],
        )
        mutype_shape = tuple([2 for _ in range(len(tc))])
        mc = mut.MutationTypeCounter(tc, mutype_shape)
        assert len(mc.labels_dict) == 7
        assert len(mc) == 12
        assert mc.rooted == False
        assert mc.phased == False
        assert mc.BranchTypeCounter == tc
        assert mc.sample_configuration == [("a", "a"), ("b", "b")]

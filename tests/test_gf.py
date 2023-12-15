import numpy as np
import pytest
import sympy

import agemo.gflib as gflib
import agemo.mutations as mut
import agemo.events as eventslib

import agemo.legacy.mutations as smut
import agemo.legacy.gflib as gfleg

import tests.gfdev as gfdev


@pytest.mark.matrix_aux
class TestMatrixAux:
    @pytest.mark.parametrize(
        "sample_list, check",
        [
            (
                ([(), ("a", "b"), ("c", "d")]),
                (
                    [
                        (1, 1, ((), ("ab",), ("c", "d"))),
                        (2, 1, ((), ("a", "b"), ("cd",))),
                    ]
                ),
            ),
            (
                ([("a", "b", "b"), (), ("a",)]),
                (
                    [
                        (0, 2, (("ab", "b"), (), ("a",))),
                        (0, 1, (("a", "bb"), (), ("a",))),
                    ]
                ),
            ),
        ],
    )
    def test_coalescence(self, sample_list, check):
        ces = eventslib.CoalescenceEventsSuite(len(sample_list))
        result = ces._single_step(sample_list)

        for test, truth in zip(result, check):
            assert all(test[i] == truth[i] for i in range(len(test)))

    @pytest.mark.parametrize(
        "sample_list, check",
        [
            (
                ([(), ("a", "b"), ("c", "d")]),
                (
                    [
                        (0, 1, ((), ("ab",), ("c", "d"))),
                        (1, 1, ((), ("a", "b"), ("cd",))),
                    ]
                ),
            ),
            (
                ([("a", "b", "b"), (), ("a",)]),
                (
                    [
                        (0, 2, (("ab", "b"), (), ("a",))),
                        (0, 1, (("a", "bb"), (), ("a",))),
                    ]
                ),
            ),
        ],
    )
    def test_coalescence_same_rates(self, sample_list, check):
        coalescence_rates = (0, 0, 1)
        ces = eventslib.CoalescenceEventsSuite(len(sample_list), coalescence_rates)
        result = ces._single_step(sample_list)
        for test, truth in zip(result, check):
            assert all(test[i] == truth[i] for i in range(len(test)))

    @pytest.mark.parametrize(
        "sample_list, check",
        [
            (
                ([(), ("a", "b"), ("c", "d")]),
                (
                    [
                        (3, 1, ((), ("b",), ("a", "c", "d"))),
                        (3, 1, ((), ("a",), ("b", "c", "d"))),
                    ]
                ),
            ),
            (
                ([(), ("a", "a"), ("b", "b")]),
                ([(3, 2, ((), ("a",), ("a", "b", "b")))]),
            ),
            (
                ([(), ("a", "a", "c", "c"), ("b", "b")]),
                (
                    [
                        (3, 2, ((), ("a", "c", "c"), ("a", "b", "b"))),
                        (3, 2, ((), ("a", "a", "c"), ("b", "b", "c"))),
                    ]
                ),
            ),
        ],
    )
    def test_migration(self, sample_list, check):
        migration_idx = 3
        mige = eventslib.MigrationEvent(migration_idx, 0, 1)
        result = mige._single_step(sample_list)
        for test, truth in zip(result, check):
            assert all(test[i] == truth[i] for i in range(len(test)))

    def test_migration_empty(self):
        sample_list = [(), (), ("c", "d")]
        migration_idx = 3
        mige = eventslib.MigrationEvent(migration_idx, 0, 1)
        result = mige._single_step(sample_list)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_exodus_empty(self):
        sample_list = [(), (), ("c", "d")]
        split_idx = 4
        pse = eventslib.PopulationSplitEvent(split_idx, 0, 1)
        result = pse._single_step(sample_list)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_exodus(self):
        sample_list = [(), ("a", "a"), ("c", "d")]
        split_idx = 4
        pse = eventslib.PopulationSplitEvent(split_idx, 0, 1, 2)
        result = pse._single_step(sample_list)
        check = [(4, 1, (("a", "a", "c", "d"), (), ()))]
        for test, truth in zip(result, check):
            assert all(test[i] == truth[i] for i in range(len(test)))


@pytest.mark.matrix_simple
class TestSimpleModels:
    def test_single_step(self):
        sample_list = [("a", "a", "b", "b")]
        branch_type_counter = mut.BranchTypeCounter(sample_list)
        gfobj = gflib.GfMatrixObject(branch_type_counter)
        multiplier_array, new_state_list = gfobj.gf_single_step(sample_list)
        expected_new_state_list = [
            (("aa", "b", "b"),),
            (("a", "ab", "b"),),
            (("a", "a", "bb"),),
        ]
        assert new_state_list == expected_new_state_list
        assert all(x == 6 for x in multiplier_array[:, 1, 0])
        assert multiplier_array[0, 0, 0] == 1
        assert multiplier_array[1, 0, 0] == 4
        assert multiplier_array[2, 0, 0] == 1
        assert np.array_equal(
            multiplier_array[0, 1, 1:], np.array([2, 2, 0, 0], dtype=np.uint8)
        )

    def test_single_step_exodus(self):
        sample_list = [("a", "a", "b", "b"), ()]
        branch_type_counter = mut.BranchTypeCounter(sample_list)
        event_list = [
            eventslib.PopulationSplitEvent(2, 1, 0),
        ]
        gfobj = gflib.GfMatrixObject(branch_type_counter, event_list)
        multiplier_array, new_state_list = gfobj.gf_single_step(sample_list)
        expected_new_state_list = [
            (("aa", "b", "b"), ()),
            (("a", "ab", "b"), ()),
            (("a", "a", "bb"), ()),
            ((), ("a", "a", "b", "b")),
        ]
        assert all(x == 1 for x in multiplier_array[:, 1, 2])
        assert new_state_list == expected_new_state_list
        assert all(x == 6 for x in multiplier_array[:, 1, 0])
        assert multiplier_array[0, 0, 0] == 1
        assert multiplier_array[1, 0, 0] == 4
        assert multiplier_array[2, 0, 0] == 1
        assert np.array_equal(
            multiplier_array[0, 1, 3:], np.array([2, 2, 0, 0], dtype=np.uint8)
        )

    def test_single_step_exodus2(self):
        sample_list = [("a", "a"), ("b", "b")]
        branch_type_counter = mut.BranchTypeCounter(sample_list)
        event_list = [eventslib.PopulationSplitEvent(2, 1, 0)]
        gfobj = gflib.GfMatrixObject(branch_type_counter, event_list)
        multiplier_array, new_state_list = gfobj.gf_single_step(sample_list)
        # expected_new_state_list = [
        #    (("aa"), ("b", "b")),
        #    (("a", "a"), ("bb")),
        #    ((), ("a", "a", "b", "b")),
        # ]
        assert np.all(
            np.array_equal(x, np.array([1, 1, 1, 2, 2, 0, 0], dtype=np.uint8))
            for x in multiplier_array[:, 1]
        )


@pytest.mark.from_matrix
class TestPaths:
    def test_paths_pre_laplace(self, return_gf):
        variables_array, (paths_mat, eq_mat), gf_original = return_gf
        self.equations_pre_laplace(eq_mat, paths_mat, variables_array, gf_original)

    def equations_pre_laplace(self, eq_mat, paths, variables_array, gf_original):
        eqs = np.zeros(len(paths), dtype=object)
        for i, path in enumerate(paths):
            ma = eq_mat[np.array(path, dtype=int)]
            eqs[i] = np.prod(gflib.equations_from_matrix(ma, variables_array))

    @pytest.fixture(
        scope="class",
        params=[
            ([(1, 2, 0)], sympy.symbols('E', real=True, positive=True), None, None),
            (None, None, [(2, 1)], sympy.symbols('M', real=True, positive=True)),
            (
                [(1, 2, 0)],
                sympy.symbols('E', real=True, positive=True),
                [(2, 1)],
                sympy.symbols('M', real=True, positive=True),
            ),
        ],
        ids=["DIV", "MIG", "IM"],
    )
    def return_gf(self, request):
        sample_list = [(), ("a", "a"), ("b", "b")]
        # ancestral_pop = 0
        coalescence_rates = (
            sympy.symbols('c0', real=True, positive=True),
            sympy.symbols('c1', real=True, positive=True),
            sympy.symbols('c2', real=True, positive=True),
        )
        coalescence_rate_idxs = (0, 1, 2)
        k_max = {"m_1": 2, "m_2": 2, "m_3": 2, "m_4": 2}
        mutype_labels, max_k = zip(*sorted(k_max.items()))
        # branchtype_dict_mat = mut.make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=mutype_labels)
        branchtype_dict_mat = gfdev.make_branchtype_dict_idxs_gimble()
        branch_type_counter = mut.BranchTypeCounter(
            sample_list, branchtype_dict=branchtype_dict_mat
        )
        branchtype_dict_chain = smut.make_branchtype_dict(
            sample_list, mapping="unrooted", labels=mutype_labels
        )
        (
            exodus_direction,
            exodus_rate,
            migration_direction,
            migration_rate,
        ) = request.param

        event_list = []
        variables_array = list(coalescence_rates)
        migration_rate_idx, exodus_rate_idx = None, None
        if migration_rate is not None:
            migration_rate_idx = len(variables_array)
            variables_array.append(migration_rate)
            event_list.append(
                eventslib.MigrationEvent(migration_rate_idx, *migration_direction[0])
            )
        if exodus_rate is not None:
            exodus_rate_idx = len(variables_array)
            variables_array.append(exodus_rate)
            *derived, ancestral = exodus_direction[0]
            event_list.append(
                eventslib.PopulationSplitEvent(exodus_rate_idx, ancestral, *derived)
            )
        m = sympy.symbols('m', real=True, positive=True)
        variables_array += [m for _ in mutype_labels]
        variables_array = np.array(variables_array, dtype=object)

        gfobj = gflib.GfMatrixObject(branch_type_counter, event_list)
        gf_mat = list(gfobj.make_gf())

        gfobj2 = gfleg.GfObject(
            sample_list,
            coalescence_rates,
            branchtype_dict_chain,
            exodus_rate=exodus_rate,
            exodus_direction=exodus_direction,
            migration_rate=migration_rate,
            migration_direction=migration_direction,
        )
        gf_original = sum(gfobj2.make_gf())

        return (variables_array, gf_mat, gf_original)

@pytest.mark.diff
@pytest.mark.collapse
class TestCollapseGraph:
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
        btc = mut.BranchTypeCounter(sample_list)
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
        ) = gfdev.collapse_graph(gfObj, graph_array, adjacency_matrix, eq_matrix)
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
        graph_array, adjacency_matrix, eq_matrix = gfdev.make_graph(gfobj)
        collapsed_graph_array, *_ = gfdev.collapse_graph(
            gfobj, graph_array, adjacency_matrix, eq_matrix
        )
        print(exp_graph_array)
        print(collapsed_graph_array)
        for o, e in zip(collapsed_graph_array, exp_graph_array):
            assert o == e

    def get_gf_no_mutations(self, sample_list):
        pse = eventslib.PopulationSplitEvent(2, 0, 1)
        btc = mut.BranchTypeCounter(sample_list, rooted=True)
        gfobj = gflib.GfMatrixObject(
            btc,
            [
                pse,
            ],
        )
        return gfobj
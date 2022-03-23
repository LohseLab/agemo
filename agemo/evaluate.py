import numpy as np
import numba

import agemo.diff as gfdiff
import agemo.mutations as gfmuts

# getting out bSFS


class GfEvaluatorLegacy:
    def __init__(self, gfobj, k_max, mutype_array):
        delta_idx = gfobj.exodus_rate  # what value if None
        # only works with single delta_idx!
        (
            self.eq_graph_array,
            eq_array,
            to_invert,
            eq_matrix,
        ) = gfobj.equations_graph()
        self.dependency_sequence = gfdiff.resolve_dependencies(
            self.eq_graph_array
        )
        self.num_branchtypes = len(k_max)
        self.final_result_shape = k_max + 2
        size = len(mutype_array)
        # marg_iterator = gfdiff.marginals_nuissance_objects(k_max)
        # slices = marg_iterator[-1]
        # prepare_graph_evaluation_with_marginals_alt(eq_matrix, to_invert_array,
        # eq_array, size, delta_idx, subsetdict, mutype_array, mutype_shape)
        self.subsetdict = gfdiff.product_subsetdict_marg(
            tuple(self.final_result_shape), mutype_array
        )
        f_tuple = gfdiff.prepare_graph_evaluation_with_marginals(
            eq_matrix,
            to_invert,
            eq_array,
            size,
            delta_idx,
            self.subsetdict,
            mutype_array,
            self.final_result_shape,
        )
        num_eq_non_inverted = np.sum(to_invert == 0)
        num_eq_tuple = (
            num_eq_non_inverted,
            to_invert.size - num_eq_non_inverted,
        )
        self.evaluator = gfdiff.evaluate_single_point_with_marginals(
            size, num_eq_tuple, f_tuple
        )
        self.multiplier_matrix = gfdiff.taylor_to_probability_coeffs(
            mutype_array, self.final_result_shape, include_marginals=True
        )

    def evaluate(self, theta, var, time):
        try:
            results = self.evaluator(var, time)
        except ZeroDivisionError:
            var = self.adjust_parameters(var)
            results = self.evaluator(var, time)
        final_result_flat = gfdiff.iterate_eq_graph(
            self.dependency_sequence,
            self.eq_graph_array,
            results,
            self.subsetdict,
        )
        theta_multiplier_matrix = gfdiff.taylor_to_probability(
            self.multiplier_matrix, theta
        )
        no_marginals = (theta_multiplier_matrix * final_result_flat).reshape(
            self.final_result_shape
        )

        # filling of array needed here!!!
        # final_result = np.zeros(self.final_result_shape, dtype=np.float64)

        return gfmuts.adjust_marginals_array(
            no_marginals, self.num_branchtypes
        )

    def adjust_parameters(self, var, factor=1e-5):
        epsilon = (
            np.random.randint(
                low=-100, high=100, size=len(var) - self.num_branchtypes
            )
            * factor
        )
        var[: -self.num_branchtypes] += epsilon
        return var


class BSFSEvaluator:
    def __init__(self, gfObj, MutationTypeCounter):
        if MutationTypeCounter.phased:
            raise NotImplementedError('Calculating the bSFS for the fully phased case is still under development.')
        num_discrete_events = len(gfObj.discrete_events) 
        if num_discrete_events==0:
            delta_idx = None
        elif num_discrete_events==1:
            delta_idx = gfObj.discrete_events[0]
        else:
            raise NotImplementedError('BSFSEvaluator can only deal with 1 discrete event.')
        
        # only works with single delta_idx!
        (
            self.eq_graph_array,
            eq_array,
            to_invert,
            eq_matrix,
        ) = gfObj.equations_graph()
        self.dependency_sequence = gfdiff.resolve_dependencies(
            self.eq_graph_array
        )
        self.eq_graph_array = numba.typed.List(self.eq_graph_array)
        
        self.final_result_shape = MutationTypeCounter.mutype_shape
        size, self.num_branchtypes = MutationTypeCounter.all_mutypes.shape
        self.simple_reshape = np.prod(self.final_result_shape) == size
        self.subsetdict = self.make_product_subsetdict(MutationTypeCounter)
        self.all_mutypes_ravel = MutationTypeCounter.all_mutypes_ravel
        # final step:
        # eventually also: mapping mutypes with same probability (part of MutationTypeCounter obj)

        f_tuple = gfdiff.prepare_graph_evaluation_with_marginals(
            eq_matrix,
            to_invert,
            eq_array,
            size,
            delta_idx,
            self.subsetdict,
            MutationTypeCounter.all_mutypes,
            self.final_result_shape,
        )
        num_eq_non_inverted = np.sum(to_invert == 0)
        num_eq_tuple = (
            num_eq_non_inverted,
            to_invert.size - num_eq_non_inverted,
        )
        self.evaluator = gfdiff.evaluate_single_point_with_marginals(
            size, num_eq_tuple, f_tuple
        )
        self.multiplier_matrix = gfdiff.taylor_to_probability_coeffs(
            MutationTypeCounter.all_mutypes,
            self.final_result_shape,
            include_marginals=True,
        )

    def evaluate(self, theta, var, time):
        try:
            results = self.evaluator(var, time)
        except ZeroDivisionError:
            var = self.adjust_parameters(var)
            results = self.evaluator(var, time)
        final_result_flat = gfdiff.iterate_eq_graph(
            self.dependency_sequence,
            self.eq_graph_array,
            results,
            self.subsetdict,
        )
        theta_multiplier_matrix = gfdiff.taylor_to_probability(
            self.multiplier_matrix, theta
        )
        no_marginals = theta_multiplier_matrix * final_result_flat

        if self.simple_reshape:
            final_result = no_marginals.reshape(self.final_result_shape)
        else:
            final_result = np.zeros(self.final_result_shape, dtype=np.float64)
            final_result.flat[self.all_mutypes_ravel] = no_marginals
        return gfmuts.adjust_marginals_array(
            final_result, self.num_branchtypes
        )

    def adjust_parameters(self, var, factor=1e-5):
        epsilon = (
            np.random.randint(
                low=-100, high=100, size=len(var) - self.num_branchtypes
            )
            * factor
        )
        var[: -self.num_branchtypes] += epsilon
        return var

    def make_product_subsetdict(self, MutationTypeCounter):
        product_subsetdict_with_gaps = gfdiff.product_subsetdict_marg(
            self.final_result_shape, MutationTypeCounter.all_mutypes
        )
        if self.simple_reshape:
            return product_subsetdict_with_gaps
        else:
            reverse_mapping = {
                value: idx
                for idx, value in enumerate(
                    MutationTypeCounter.all_mutypes_ravel
                )
            }
            product_subsetdict_no_gaps = numba.typed.Dict()
            for key, value in product_subsetdict_with_gaps.items():
                product_subsetdict_no_gaps[reverse_mapping[key]] = np.array(
                    [reverse_mapping[v] for v in value], dtype=np.uint64
                )
            return product_subsetdict_no_gaps

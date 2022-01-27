import itertools
import numpy as np
import pytest
import sage.all
import scipy.stats as stats

from scipy.special import factorial

import gf.gflib as gflib
import gf.diff as gfdiff
import gf.inverse as gfinverse
import gf.mutations as gfmuts
import gf.evaluate as gfeval
import gf.legacy.inverse as linverse
import gf.legacy.mutations as smuts
import tests.gfdev as gfdev

@pytest.mark.taylor
class Test_taylor_series_coefficients:
	def test_taylor_polynomial(self):
		denom = np.arange(1,5, dtype=int)
		var_array = np.arange(1,5, dtype=float)/10
		diff_array = (2,2)
		num_mutypes = 2
		dot_product = gfdiff.simple_dot_product(denom, var_array)
		result = gfdiff.taylor_coeff_inverse_polynomial(denom, var_array, diff_array, num_mutypes, dot_product)
		expected = 3.55556
		assert np.isclose(expected, result)

	def test_taylor_exponential(self):
		denom = np.arange(1,5, dtype=int)
		var_array = np.arange(1,5, dtype=float)/10
		diff_array = (2,2)
		num_mutypes = 2
		time = 2
		exponential_part = np.exp(-2*np.dot(var_array, denom))
		result = gfdiff.taylor_coeff_exponential(-time, denom, exponential_part, diff_array, num_mutypes)
		expected = 1.42776
		assert np.isclose(expected, result)

	def test_prod_of_polynomials(self):
		eq_matrix = np.array([
			[[1,0,0,0],[1,1,1,1]],
			[[0,1,0,0],[1,1,1,1]],
			[[1,0,0,0],[1,0,1,0]]],
			dtype=np.int8)
		shape = (2,2)
		var_array = np.array([0.1,0.2,0.3,0.3], dtype=np.float64)
		symbolic_var_array = np.array(
			[sage.all.Rational(0.1), sage.all.Rational(0.2), sage.all.SR.var('m1'), sage.all.SR.var('m2')], 
			dtype=object
			)
		result = gfdiff.compile_non_inverted_eq(eq_matrix, shape, shape)(var_array)
		subsetdict = gfdiff.product_subsetdict(shape)
		combined_result = gfdiff.product_f(subsetdict, result)
		#from symbolic eq
		subs_dict = {b:var_array[-1] for b in symbolic_var_array[-2:]}
		symbolic_eq = np.prod(gflib.equations_from_matrix(eq_matrix, symbolic_var_array))
		expected = return_symbolic_result(symbolic_eq, subs_dict, symbolic_var_array[-2:], shape)
		assert np.allclose(expected, combined_result)

	@pytest.mark.parametrize("eq_matrix, delta_in_nom",
		[(np.array([
			[[1,0,0,0],[1,1,1,1]],
			[[0,1,0,0],[2,1,0,2]],
			[[1,0,0,0],[1,1,1,0]]],
			dtype=np.int8), True),
		(np.array([
			[[1,0,0,0],[1,1,1,1]],
			[[1,0,0,0],[2,1,0,2]],
			[[1,0,0,0],[1,1,1,0]]],
			dtype=np.int8), False)])
	def test_diff_inverse_laplace(self, eq_matrix, delta_in_nom):
		dummy_variable_idx = 1
		dummy_variable = sage.all.SR.var('E')
		eq_matrix_no_delta = np.delete(eq_matrix, dummy_variable_idx, axis=2)
		shape = (2,2)
		var_array = np.array([0.1,0.3,0.3], dtype=np.float64)
		symbolic_var_array = np.array(
			[sage.all.Rational(0.1), dummy_variable, sage.all.SR.var('m1'), sage.all.SR.var('m2')], 
			dtype=object
			)
		time = 1.0
		subsetdict = gfdiff.product_subsetdict(shape)
		result = gfdiff.compile_inverted_eq(eq_matrix_no_delta, shape, subsetdict, delta_in_nom, shape)(var_array, time)
		print(result)
		#from symbolic eq
		subs_dict = {b:var_array[-1] for b in symbolic_var_array[-2:]}
		subs_dict[sage.all.SR.var('T')] = time
		symbolic_eq = np.prod(gflib.equations_from_matrix(eq_matrix, symbolic_var_array))
		inverted_symbolic_eq = linverse.return_inverse_laplace(symbolic_eq, dummy_variable)
		expected = return_symbolic_result(inverted_symbolic_eq, subs_dict, symbolic_var_array[-2:], shape)
		assert np.allclose(expected, result)

	@pytest.mark.parametrize("eq_matrix, delta_in_denom, delta_in_nom",
		[(np.array([
		[[1,0,0,0],[1,0,1,1]],
		[[0,1,0,0],[2,1,0,2]],
		[[1,0,0,0],[1,0,1,0]]],
		dtype=np.int8), np.array([False, True, False], dtype=bool), True),
		(np.array([
			[[1,0,0,0],[1,0,1,1]],
			[[1,0,0,0],[2,1,0,2]],
			[[1,0,0,0],[1,0,1,0]]],
			dtype=np.int8), np.array([False, True, False], dtype=bool), False)
		])
	def test_diff_inverse_laplace2(self, eq_matrix, delta_in_denom, delta_in_nom):
		#eq in which some factors have dummy variable, others don't
		dummy_variable_idx = 1
		dummy_variable = sage.all.SR.var('E')
		eq_matrix_no_delta = np.delete(eq_matrix, dummy_variable_idx, axis=2)
		shape = (2,2)
		var_array = np.array([0.1,0.3,0.3], dtype=np.float64)
		symbolic_var_array = np.array(
			[sage.all.Rational(0.1), dummy_variable, sage.all.SR.var('m1'), sage.all.SR.var('m2')], 
			dtype=object
			)
		time = 1.0
		subsetdict = gfdiff.product_subsetdict(shape)
		result_inverted_part = gfdiff.compile_inverted_eq(eq_matrix_no_delta[delta_in_denom], shape, subsetdict, delta_in_nom, shape)(var_array, time)
		result_non_inverted_part = gfdiff.compile_non_inverted_eq(eq_matrix_no_delta[~delta_in_denom], shape, shape)(var_array)
		result = gfdiff.product_f(subsetdict, np.vstack((result_inverted_part[None, :], result_non_inverted_part)))
		print('result')
		print(result)
		#from symbolic eq
		subs_dict = {b:var_array[-1] for b in symbolic_var_array[-2:]}
		subs_dict[sage.all.SR.var('T')] = time
		symbolic_eq = np.prod(gflib.equations_from_matrix(eq_matrix, symbolic_var_array))
		inverted_symbolic_eq = linverse.return_inverse_laplace(symbolic_eq, dummy_variable)
		expected = return_symbolic_result(inverted_symbolic_eq, subs_dict, symbolic_var_array[-2:], shape)
		print('expected')
		print(expected)
		assert np.allclose(expected, result)

def return_symbolic_result(eq, subs_dict, branchtypes, shape, root=(0,0)):
	symbolic_diff = np.zeros(shape, dtype=np.float64)
	for mutype in itertools.product(*(range(i) for i in shape)):
		if mutype == root:
			symbolic_diff[mutype] = eq.subs(subs_dict)
		else:
			how_to_diff = smuts.single_partial(branchtypes, mutype)
			symbolic_diff[mutype] = 1/np.prod(factorial(mutype)) * sage.all.diff(eq, *how_to_diff).subs(subs_dict)
			
	return symbolic_diff

@pytest.mark.collapse
class Test_collapse_graph:
	def test_collapse_graph(self):
		graph_array = ((1,2,4,5),(2,),(3,),(6,),(3,),(3,),tuple())
		eq_matrix = np.array([
			[[0,0],[0,1]],
			[[0,1],[0,1]],
			[[0,0],[0,0]]],
			dtype=np.uint8
			)
		adjacency_matrix = np.full((len(graph_array), len(graph_array)), fill_value=255, dtype=np.int8)
		adjacency_matrix[np.array([0,0,1]), np.array([1,2,2])] = 0
		adjacency_matrix[np.array([2]), np.array([3])] = 1
		adjacency_matrix[np.array([0,0,3,4,5]), np.array([4,5,6,3,3])] = 2
		sample_list = [('a', 'a'),('b', 'b')]
		coalescence_rates = (0,1)
		branchtype_dict = {}
		exodus_rate = 1
		exodus_direction = [(1,0)]
		gfObj = gflib.GFMatrixObject(sample_list, coalescence_rates, branchtype_dict, exodus_rate=exodus_rate, exodus_direction=exodus_direction)
		collapsed_graph_array, adjacency_matrix_b, eq_array, to_invert_array = gfObj.collapse_graph(graph_array, adjacency_matrix, eq_matrix)
		expected_graph_array= ((1, 2, 5, 6), (3,), (3,), (4,), (), (4,), (4,))
		expected_to_invert_array = np.zeros(9, dtype=bool)
		expected_to_invert_array[-2:] = 1
		expected_eq_array = ((2,), (2,), (2,), (2,), (2,), (2,), (2,), (0, 1), (0, 0, 1))

		def compare(ar1, ar2):
			for a, b in zip(ar1, ar2):
				assert np.array_equal(np.array(a), np.array(b))

		compare(expected_graph_array, collapsed_graph_array)
		compare(expected_eq_array, eq_array)
		assert np.array_equal(expected_to_invert_array, to_invert_array)

	@pytest.mark.parametrize(
		'sample_list, k_max, branchtype_dict, exp_graph_array',
		[
		([(), ('a', 'a')], {'m_1':2}, {'a':0}, [(1,3), (2,), (),()]),
		([(), ('a', 'a', 'a')], {'m_1':2, 'm_2':2}, {'a':0, 'aa':1}, [(1, 4, 5), (2,), (3,), (), (3,), ()]),
		])
	def test_graph_with_multiple_endpoints(self, sample_list, k_max, branchtype_dict, exp_graph_array):
		gfobj = self.get_gf_no_mutations(sample_list, k_max, branchtype_dict)
		delta_idx = gfobj.exodus_rate
		graph_array, adjacency_matrix, eq_matrix = gfobj.make_graph()
		collapsed_graph_array, *_ = gfobj.collapse_graph(graph_array, adjacency_matrix, eq_matrix)		
		print(exp_graph_array)
		print(collapsed_graph_array)
		for o,e in zip(collapsed_graph_array, exp_graph_array):
			assert o==e

	def get_gf_no_mutations(self, sample_list, k_max, branchtype_dict):
		coalescence_rate_idxs = (0, 1)
		exodus_rate_idx = 2 
		exodus_direction = [(1,0),]
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		gfobj = gflib.GFMatrixObject(
			sample_list, 
			coalescence_rate_idxs, 
			branchtype_dict,
			exodus_direction=exodus_direction,
			exodus_rate=exodus_rate_idx
			)
		return gfobj

@pytest.mark.taylor2
class Test_taylor2:
	@pytest.mark.parametrize('size', [2, 3])
	def test_combining_probabilities(self, size):
		gfobj = self.get_gf_no_mutations(size)
		max_k = np.full(size-1,fill_value=2, dtype=int)
		shape = tuple(max_k+1)
		variable_array = np.array([1., 2.], dtype=np.float64)
		theta = .5
		theta_array = np.full(size-1, fill_value=theta)
		variable_array = np.hstack((variable_array, theta_array))
		time = 1.5
		ordered_mutype_list = [sage.all.SR.var(f'm_{idx}') for idx in range(1,size)]
		num_mutypes = len(ordered_mutype_list)
		alt_variable_array = np.hstack((variable_array[:2], np.array(ordered_mutype_list)))
		result_with_marginals = evaluate_graph_marginals(gfobj, max_k, theta, variable_array, time, False)
		print(result_with_marginals)
		exp_result = evaluate_symbolic_equation(gfobj, ordered_mutype_list, max_k, theta, alt_variable_array, time)
		assert np.allclose(exp_result, result_with_marginals)

	def test_IM_models(self, get_IM_gfobject):
		gfobj, parameter_combo, model, adjust_marginals = get_IM_gfobject
		num_variables = gfobj.num_variables if gfobj.exodus_rate is None else gfobj.num_variables-1
		max_k = np.array([2,2,2,2], dtype=int)
		ordered_mutype_list = [sage.all.SR.var(f'm_{idx}') for idx in range(1,len(max_k)+1)]
		shape = tuple(max_k+1)
		#variables depending on model: c0, c1, c2, M, E
		theta, variable_array, time = parameter_combo
		theta_array = np.full(len(max_k), fill_value=theta)
		var = np.hstack((variable_array, theta_array))
		#var_symbolic = np.hstack((variable_array, ordered_mutype_list))
		#if gfobj.exodus_rate is not None:
		#	var_sage = np.zeros(gfobj.num_variables, dtype=object)
		#	var_sage[:-1] = [sage.all.Rational(v) for v in variable_array]
		#	var_sage[-1] = sage.all.SR('E')
		#	var_symbolic = np.hstack((var_sage, ordered_mutype_list))
		#else:
		#	var_symbolic = np.hstack((variable_array, ordered_mutype_list))
			
		result_with_marginals = evaluate_graph_marginals(gfobj, max_k, theta, var, time, adjust_marginals)
		self.compare_ETPs_model(model, result_with_marginals)
		
	def get_gf_no_mutations(self, size):
		sample_list = [(), ('a',)*size]
		coalescence_rate_idxs = (0, 1)
		exodus_rate_idx = 2 
		exodus_direction = [(1,0),]
		k_max = {f'm_{idx}':2 for idx in range(1,size)}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict_mat = {'a'*idx:idx-1 for idx in range(1,size)} 
		gfobj = gflib.GFMatrixObject(
			sample_list, 
			coalescence_rate_idxs, 
			branchtype_dict_mat,
			exodus_direction=exodus_direction,
			exodus_rate=exodus_rate_idx
			)
		return gfobj

	#def evaluate_graph_marginals(self, gfobj, k_max, theta, var, time):
	#	gfEvalObj = gfeval.gfEvaluator(gfobj, k_max)
	#	return gfEvalObj.evaluate(theta, var, time)

	def compare_ETPs_model(self, model, ETPs):
		precalc_ETPs = np.squeeze(np.load(f'tests/ETPs/{model}.npy'))
		#np.save('IM_test.npy', ETPs)
		assert np.allclose(precalc_ETPs, ETPs)

	@pytest.fixture(
	scope='class', 
	params=[
		([(1,2,0)], sage.all.SR.var('E'), None, None,[0.51, np.array([.1, .2, .3], dtype=np.float64), 1.5], 'DIV_taylor2', False),
		([(1,2,0)], sage.all.SR.var('E'), None, None,[72/125, np.array([1.0, 15/13, 5/2], dtype=np.float64), 10/3], 'DIV', True),
		(None, None, [(2,1)], sage.all.SR.var('M'), [0.51, np.array([.1, .2, .3, .4], dtype=np.float64), 1.5], 'MIG_BA_taylor2', False),
		(None, None, [(2,1)], sage.all.SR.var('M'), [312/625, np.array([0.0, 1.0, 13/6, 134369693800271/73829502088061], dtype=np.float64), 0.0],'MIG_BA', True),
		([(1,2,0)], sage.all.SR.var('E'), [(2,1)], sage.all.SR.var('M'), [0.51, np.array([1.0, 0.5, 0.9, 0.001], dtype=np.float64), 1.5], 'IM_BA_taylor2', False),
		([(1,2,0)], sage.all.SR.var('E'), [(1,2)], sage.all.SR.var('M'), [72/125, np.array([1.0, 15/13, 5/2, 21/10], dtype=np.float64), 10/3], 'IM_AB', True),
		],
	ids=[
		'DIV_taylor2',
		'DIV',
		'MIG_taylor2',
		'MIG', 
		'IM_taylor2',
		'IM'
		],
	)
	def get_IM_gfobject(self, request):
		return get_IM_gfobject(request.param)


@pytest.mark.epsilon
class Test_epsilon:
	def test_IM_epsilon(self):
		params = ([(1,2,0)], sage.all.SR.var('E'), [(2,1)], sage.all.SR.var('M'), [0.5, np.array([1.0, .2, .3, .4], dtype=np.float64), 1.5], 'IM_BA_zerodivision', True)
		gfobj, parameter_combo, model, adjust_marginals = get_IM_gfobject(params)
		num_variables = gfobj.num_variables if gfobj.exodus_rate is None else gfobj.num_variables-1
		max_k = np.array([2,2,2,2], dtype=int)
		ordered_mutype_list = [sage.all.SR.var(f'm_{idx}') for idx in range(1,len(max_k)+1)]
		shape = tuple(max_k+1)
		#variables depending on model: c0, c1, c2, M, E
		theta, variable_array, time = parameter_combo
		#factor = 1e-5
		#epsilon = 1. + np.random.randint(low=-100, high=100, size=len(variable_array)) * factor
		#variable_array*=epsilon
		#print(variable_array)
		theta_array = np.full(len(max_k), fill_value=theta)
		var = np.hstack((variable_array, theta_array))
		
		result_with_marginals = evaluate_graph_marginals(gfobj, max_k, theta, var, time, adjust_marginals)
		sim_counts = np.load('tests/ETPs/IM_BA_epsilon_sim.npy')
		result_with_marginals[result_with_marginals<0] = 0
		expected_counts = result_with_marginals * np.sum(sim_counts)
		chisquare(sim_counts, expected_counts)

def evaluate_graph_marginals(gfobj, k_max, theta, var, time, adjust_marginals):
	gfEvalObj = gfeval.gfEvaluator(gfobj, k_max)
	if adjust_marginals:
		return gfEvalObj.evaluate(theta, var, time)
	else:
		return evaluate_no_marginal(gfEvalObj, theta, var, time)

def evaluate_symbolic_equation(gfobj, ordered_mutype_list, max_k, theta, var, time, sage_inverse=False):
	theta = sage.all.Rational(theta)
	rate_dict = {b:theta for b in ordered_mutype_list}
	paths, eq_matrix = gfobj.make_gf()
	if not sage_inverse:
		alt_eqs = gfdev.equations_from_matrix_with_inverse(eq_matrix, paths, var, time, gfobj.exodus_rate)
	else:
		alt_eqs = gfdev.equations_with_sage(eq_matrix, paths, var, sage.all.Rational(time), gfobj.exodus_rate)
	gf_alt = sum(alt_eqs)
	result = smuts.depth_first_mutypes(max_k, ordered_mutype_list, gf_alt, theta, rate_dict)
	return result.astype(np.float64)		

def get_IM_gfobject(params):
		sample_list = [(),('a','a'),('b','b')]
		ancestral_pop = 0
		coalescence_rates = (sage.all.SR.var('c0'), sage.all.SR.var('c1'), sage.all.SR.var('c2'))
		coalescence_rate_idxs = (0, 1, 2)
		k_max = {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		#branchtype_dict_mat = gfmuts.make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=mutype_labels)
		branchtype_dict_mat = gfdev.make_branchtype_dict_idxs_gimble()
		exodus_direction, exodus_rate, migration_direction, migration_rate, variable_array, model, adjust_marginals = params
		
		variables_array = list(coalescence_rates)
		migration_rate_idx, exodus_rate_idx = None, None
		if migration_rate!=None:
			migration_rate_idx = len(variables_array)
			variables_array.append(migration_rate)
		if exodus_rate!=None:
			exodus_rate_idx = len(variables_array)
			variables_array.append(exodus_rate)
		variables_array += [sage.all.SR.var(m) for m in mutype_labels]
		variables_array = np.array(variables_array, dtype=object)
		
		gfobj = gflib.GFMatrixObject(
			sample_list,
			coalescence_rate_idxs, 
			branchtype_dict_mat,
			exodus_rate=exodus_rate_idx,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate_idx,
			migration_direction=migration_direction
			)

		return gfobj, variable_array, model, adjust_marginals

def chisquare(observed, expected, p_th=0.05, recombination=False, all_sims=False):
	#expected counts need to be larger than 5 for chisquare
	#combining bins with less than 5 counts into one bin.
	obs = np.reshape(observed, -1)
	exp = np.reshape(expected, -1)
	if not (recombination or all_sims):
		assert np.all(obs[exp == 0] == 0), "chisquare error: at least one cell with expected frequency 0 has observed values"
	#bin all values with counts smaller than 5
	binning_idxs = np.digitize(exp, np.array([5]), right=True)
	exp_smaller = np.sum(exp[binning_idxs==0])
	obs_smaller = np.sum(obs[binning_idxs==0])
	exp_binned = exp[binning_idxs==1]
	obs_binned = obs[binning_idxs==1]
	not_zeros = exp_binned > 0
	if sum(not_zeros)<1:
		assert False #expected probabilities are all 0
	else:
		chisq = stats.chisquare(obs_binned[not_zeros], exp_binned[not_zeros])
		print("chisquare value:", chisq)
		print("exp", exp_binned)
		print("obs", obs_binned)
		assert chisq.pvalue > p_th

def evaluate_no_marginal(gfEvaluatorObj, theta, var, time):
	results = gfEvaluatorObj.evaluator(var, time)
	final_result = gfdiff.iterate_eq_graph(gfEvaluatorObj.dependency_sequence, gfEvaluatorObj.eq_graph_array, results, gfEvaluatorObj.subsetdict)
	theta_multiplier_matrix = gfdiff.taylor_to_probability(gfEvaluatorObj.multiplier_matrix, theta)
	return theta_multiplier_matrix * final_result
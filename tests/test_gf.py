import numpy as np
import pytest
import sage.all

import gf.gflib as gflib
import gf.mutations as mut
import gf.legacy.mutations as smut

############### testing non-matrix representation of gf #########

@pytest.mark.aux
class Test_aux:
	@pytest.mark.parametrize("input_lineages, to_join, expected",
		[(('a', 'b', 'c', 'a'), ('a', 'b'),('a', 'ab', 'c')),
		(('a', 'b', 'b'), ('b', 'b'), ('a', 'bb')),
		(('ab', 'ab'), ('ab', 'ab'), ('aabb',))])
	def test_coalesce_lineages(self, input_lineages, to_join, expected):
		output_lineages = gflib.coalesce_lineages(input_lineages, to_join)
		assert sorted(output_lineages) == sorted(expected)

	@pytest.mark.parametrize("sample_list, check",
		[(([(), ('a','b'),('c','d')]),([(1, ((), ('ab',), ('c', 'd'))), (1, ((), ('a', 'b'), ('cd',)))])),
		(([('a', 'b', 'b'),(), ('a',)]),([(2, (('ab', 'b'), (), ('a',))), (1, (('a', 'bb'), (), ('a',)))]))])
	def test_coalescence(self, sample_list, check):
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, exodus_rate=1, exodus_direction=[(1,2,0)])
		result = list(gfobj.coalescence_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	@pytest.mark.parametrize("sample_list, check",
	[(([(), ('a','b'),('c','d')]),([(sage.all.var('c1'), ((), ('ab',), ('c', 'd'))), (sage.all.SR.var('c2'), ((), ('a', 'b'), ('cd',)))])),
	(([('a', 'b', 'b'),(), ('a',)]),([(2*sage.all.var('c0'), (('ab', 'b'), (), ('a',))), (1*sage.all.SR.var('c0'), (('a', 'bb'), (), ('a',)))]))])
	def test_coalescence_rates(self, sample_list, check):
		coalescence_rates = (sage.all.SR.var('c0'), sage.all.SR.var('c1'), sage.all.SR.var('c2'))
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, coalescence_rates, branchtype_dict, exodus_rate=1, exodus_direction=[(1,2,0)])
		result = list(gfobj.coalescence_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	@pytest.mark.parametrize("sample_list, check",
		[(([(), ('a','b'),('c','d')]),([(1, ((), ('b',), ('a','c', 'd'))), (1, ((), ('a',), ('b', 'c', 'd')))])),
		(([(), ('a','a'),('b','b')]),([(2, ((), ('a',), ('a', 'b', 'b')))])),
		(([(), ('a','a', 'c', 'c'),('b','b')]),([(2, ((), ('a','c', 'c'), ('a', 'b', 'b'))),(2, ((), ('a','a','c'), ('b', 'b', 'c')))]))]
		)
	def test_migration(self, sample_list, check):
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=1, migration_direction=[(1,2)])
		result = list(gfobj.migration_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	def test_migration_empty(self):
		sample_list = [(), (),('c','d')]
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=1, migration_direction=[(1,2)])
		result = list(gfobj.migration_events(gfobj.sample_list))
		print('result:', result)
		assert isinstance(result, list)
		assert len(result)==0
		
	def test_exodus_empty(self):
		sample_list = [(), (),('c','d')]
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=1, migration_direction=[(1,2)])
		result = list(gfobj.exodus_events(gfobj.sample_list))
		print('result:', result)
		assert isinstance(result, list)
		assert len(result)==0

	def test_exodus(self):
		sample_list = [(), ('a','a'),('c','d')]
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='label')
		exodus_rate = sage.all.SR.var('E')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, exodus_rate=exodus_rate, exodus_direction=[(1,2,0)])
		result = list(gfobj.exodus_events(gfobj.sample_list))
		check = [(exodus_rate, (('a', 'a', 'c', 'd'), (), ()))]
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

@pytest.mark.gf
class Test_gf:
	def test_gf_unrooted(self):
		sample_list = [('a','a','b','b')]
		branchtype_dict = smut.make_branchtype_dict(sample_list, mapping='unrooted')
		gfobj = gflib.GFObject(sample_list, (1,), branchtype_dict)
		result = list(gfobj.make_gf())
		subs_dict = {k:0 for k in set(branchtype_dict.values())}
		assert sum([x.substitute(subs_dict) for x in result])==1

################### testing matrix representation ###################

@pytest.mark.matrix_aux
class Test_matrix_aux:
	@pytest.mark.parametrize("sample_list, check",
		[(([(), ('a','b'),('c','d')]),([(1, 1, ((), ('ab',), ('c', 'd'))), (2, 1, ((), ('a', 'b'), ('cd',)))])),
		(([('a', 'b', 'b'),(), ('a',)]),([(0, 2, (('ab', 'b'), (), ('a',))), (0 ,1, (('a', 'bb'), (), ('a',)))]))])
	def test_coalescence(self, sample_list, check):
		#rate_array: [c0,c1,c2,M,E,m_1,m_2,m_3,m_4]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='label', starting_index=5)
		gfobj = gflib.GFMatrixObject(sample_list, (0,1,2), branchtype_dict, exodus_rate=4, exodus_direction=[(1,2,0)])
		result = list(gfobj.coalescence_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	@pytest.mark.parametrize("sample_list, check",
	[(([(), ('a','b'),('c','d')]),([(0, 1, ((), ('ab',), ('c', 'd'))), (1, 1, ((), ('a', 'b'), ('cd',)))])),
	(([('a', 'b', 'b'),(), ('a',)]),([(0, 2, (('ab', 'b'), (), ('a',))), (0, 1, (('a', 'bb'), (), ('a',)))]))])
	def test_coalescence_same_rates(self, sample_list, check):
		coalescence_rates = (0, 0, 1)
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='label', starting_index=5)
		gfobj = gflib.GFMatrixObject(sample_list, coalescence_rates, branchtype_dict, exodus_rate=4, exodus_direction=[(1,2,0)])
		result = list(gfobj.coalescence_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	@pytest.mark.parametrize("sample_list, check",
		[(([(), ('a','b'),('c','d')]),([(3, 1, ((), ('b',), ('a','c', 'd'))), (3, 1, ((), ('a',), ('b', 'c', 'd')))])),
		(([(), ('a','a'),('b','b')]),([(3, 2, ((), ('a',), ('a', 'b', 'b')))])),
		(([(), ('a','a', 'c', 'c'),('b','b')]),([(3, 2, ((), ('a','c', 'c'), ('a', 'b', 'b'))),(3, 2, ((), ('a','a','c'), ('b', 'b', 'c')))]))]
		)
	def test_migration(self, sample_list, check):
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='label')
		gfobj = gflib.GFMatrixObject(sample_list, (0,1,2), branchtype_dict, migration_rate=3, migration_direction=[(1,2)])
		result = list(gfobj.migration_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	def test_migration_empty(self):
		sample_list = [(), (),('c','d')]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='label')
		gfobj = gflib.GFMatrixObject(sample_list, (0,1,2), branchtype_dict, migration_rate=3, migration_direction=[(1,2)])
		result = list(gfobj.migration_events(gfobj.sample_list))
		print('result:', result)
		assert isinstance(result, list)
		assert len(result)==0
		
	def test_exodus_empty(self):
		sample_list = [(), (),('c','d')]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='label')
		gfobj = gflib.GFMatrixObject(sample_list, (0,1,2), branchtype_dict, exodus_rate=4, exodus_direction=[(0,1)])
		result = list(gfobj.exodus_events(gfobj.sample_list))
		print('result:', result)
		assert isinstance(result, list)
		assert len(result)==0

	def test_exodus(self):
		sample_list = [(), ('a','a'),('c','d')]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='label')
		gfobj = gflib.GFMatrixObject(sample_list, (0,1,2), branchtype_dict, exodus_rate=4, exodus_direction=[(1,2,0)])
		result = list(gfobj.exodus_events(gfobj.sample_list))
		check = [(4, 1, (('a', 'a', 'c', 'd'), (), ()))]
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

@pytest.mark.matrix_simple
class Test_Simple_Models:
	def test_single_step(self):
		sample_list = [('a','a', 'b', 'b')]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='unrooted')
		gfobj = gflib.GFMatrixObject(sample_list, (0,), branchtype_dict)
		multiplier_array, new_state_list = gfobj.gf_single_step(sample_list)
		print('new_state_list:', new_state_list)
		print('multiplier_array:')
		print(multiplier_array)
		expected_new_state_list = [(('aa', 'b', 'b'),), (('a', 'ab', 'b'),), (('a', 'a', 'bb'),)]
		assert new_state_list == expected_new_state_list
		assert all(x==6 for x in multiplier_array[:,1,0])
		assert multiplier_array[0,0,0]==1
		assert multiplier_array[1,0,0]==4
		assert multiplier_array[2,0,0]==1
		assert np.array_equal(multiplier_array[0,1,1:],np.array([2,2,0,0],dtype=np.uint8))

	def test_single_step_exodus(self):
		sample_list = [('a','a', 'b', 'b'),()]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='unrooted')
		#rate_array = [c0, c1, E, m_1, m_2, m_3, m_4]
		gfobj = gflib.GFMatrixObject(sample_list, (0, 1), branchtype_dict, exodus_rate=2, exodus_direction=[(0,1),])
		multiplier_array, new_state_list = gfobj.gf_single_step(sample_list)
		print('new_state_list:', new_state_list)
		print('multiplier_array:')
		print(multiplier_array)
		expected_new_state_list = [(('aa', 'b', 'b'),()), (('a', 'ab', 'b'),()), (('a', 'a', 'bb'),()), ((),('a', 'a', 'b', 'b'))]
		assert all(x==1 for x in multiplier_array[:,1,2])
		assert new_state_list == expected_new_state_list
		assert all(x==6 for x in multiplier_array[:,1,0])
		assert multiplier_array[0,0,0]==1
		assert multiplier_array[1,0,0]==4
		assert multiplier_array[2,0,0]==1
		assert np.array_equal(multiplier_array[0,1,3:],np.array([2,2,0,0],dtype=np.uint8))

	def test_single_step_exodus2(self):
		sample_list = [('a','a'),('b', 'b')]
		branchtype_dict = mut.make_branchtype_dict_idxs(sample_list, mapping='unrooted')
		#rate_array = [c0, c1, E, m_1, m_2, m_3, m_4]
		gfobj = gflib.GFMatrixObject(sample_list, (0, 1), branchtype_dict, exodus_rate=2, exodus_direction=[(0,1),])
		multiplier_array, new_state_list = gfobj.gf_single_step(sample_list)
		print('new_state_list:', new_state_list)
		print('multiplier_array:')
		print(multiplier_array)
		expected_new_state_list = [(('aa'),('b', 'b')), (('a', 'a'),('bb')), ((),('a', 'a', 'b', 'b'))]
		assert np.all(np.array_equal(x,np.array([1,1,1,2,2,0,0],dtype=np.uint8)) for x in multiplier_array[:,1])

@pytest.mark.from_matrix
class Test_Paths:

	def test_paths_pre_laplace(self, return_gf):
		variables_array, (paths_mat, eq_mat), gf_original = return_gf	
		self.equations_pre_laplace(eq_mat, paths_mat, variables_array, gf_original)

	def equations_pre_laplace(self, eq_mat, paths, variables_array, gf_original):
		print(paths)
		print(eq_mat.shape)
		eqs = np.zeros(len(paths), dtype=object)
		for i, path in enumerate(paths):
			ma = eq_mat[np.array(path, dtype=int)]
			eqs[i] = np.prod(gflib.equations_from_matrix(ma, variables_array))
		gf_from_paths = sum(eqs)

	@pytest.fixture(
		scope='class', 
		params=[
			([(1,2,0)], sage.all.SR.var('E'), None, None),
			(None, None, [(2,1)], sage.all.SR.var('M')),
			([(1,2,0)], sage.all.SR.var('E'), [(2,1)], sage.all.SR.var('M'))
			],
		ids=[
			'DIV',
			'MIG', 
			'IM'
			],
		)
	def return_gf(self, request):
		sample_list = [(),('a','a'),('b','b')]
		ancestral_pop = 0
		coalescence_rates = (sage.all.SR.var('c0'), sage.all.SR.var('c1'), sage.all.SR.var('c2'))
		coalescence_rate_idxs = (0, 1, 2)
		k_max = {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict_mat = mut.make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=mutype_labels)
		branchtype_dict_chain = smut.make_branchtype_dict(sample_list, mapping='unrooted', labels=mutype_labels)
		exodus_direction, exodus_rate, migration_direction, migration_rate = request.param
		
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
		gf_mat = list(gfobj.make_gf())
		
		gfobj2 = gflib.GFObject(
			sample_list, 
			coalescence_rates, 
			branchtype_dict_chain,
			exodus_rate=exodus_rate,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate,
			migration_direction=migration_direction
			)
		gf_original = sum(gfobj2.make_gf())

		return (variables_array, gf_mat, gf_original)
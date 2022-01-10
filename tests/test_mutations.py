import itertools
import numpy as np
import pytest
import sage.all

import gf.gflib as gflib
import gf.mutations as mut
import gf.legacy.evaluate as leval

@pytest.mark.muts
class Test_mutypetree:
	def test_incomplete_tree(self):
		all_mutypes = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),(0,0,2,2),(0,0,3,2),
			(2,1,3,0),(2,1,2,3)
			]
		)
		root = (0,0,0,0)
		max_k = (2,2,2,2)
		mutype_tree = mut.make_mutype_tree(all_mutypes, root, max_k)
		result = {
				(0,0,0,0): [(0,0,2,2),(1,0,0,0)],
				(0,0,0,2): [(0,0,3,2),],
				(1,0,0,0): [(2,1,0,0),], 
				(2,1,0,0): [(2,1,2,0), (2,1,3,0)],
				(2,1,2,0): [(2,1,2,3)]
				}
		assert result==mutype_tree

	def test_complete_tree(self):
		root = (0,0,0,0)
		max_k = (2,2,2,2)
		all_mutypes = sorted(mut.return_mutype_configs(max_k))
		mutype_tree = mut.make_mutype_tree(all_mutypes, root, max_k)
		mutype_tree_single_digit = mut.make_mutype_tree_single_digit(all_mutypes, root, max_k)
		assert mutype_tree == mutype_tree_single_digit

	def test_muts_to_gimble(self):
		gf = 1
		max_k = (2,2,2,2)
		root = (0,0,0,0)
		exclude = [(2,3),]
		restrict_to = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),(0,0,2,2),(0,0,3,2),
			(2,1,3,0),(2,1,2,3)
			]
		)
		mutypes = tuple(f'm_{idx}' for idx in range(len(max_k)))
		gfEvaluatorObj = leval.gfEvaluator(gf, max_k, mutypes, exclude=exclude, restrict_to=restrict_to)
		expected_all_mutation_configurations = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),
			(2,1,3,0),(2,1,0,0),(2,1,1,0),(2,1,2,0)
			])

		print(gfEvaluatorObj.mutype_tree)
		#types to add:
		#(0,0,0,2),(0,0,1,2),(0,0,2,2),(2,1,2,0),(2,1,2,1),(2,1,2,2),(2,1,0,0),(2,1,1,0),(2,1,2,0)
		expected_result = {
				(0,0,0,0): [(0,0,0,2),(1,0,0,0)],
				(1,0,0,0): [(2,1,0,0),], 
				(2,1,0,0): [(2,1,1,0), (2,1,3,0)],
				(2,1,1,0): [(2,1,2,0),]
				}
		assert gfEvaluatorObj.mutype_tree == expected_result

	def test_list_marginal_idxs(self):
		marginals = np.array([(0,0,3,1),(3,0,1,3)],dtype=np.uint8)
		max_k = (2,2,2,2)
		result = [mut.list_marginal_idxs(m, max_k) for m in marginals]
		expected_result = [
			[(0,0,0,1), (0,0,1,1),(0,0,2,1)],
			[(0,0,1,0),(0,0,1,1),(0,0,1,2),(1,0,1,0),(1,0,1,1),(1,0,1,2),(2,0,1,0),(2,0,1,1),(2,0,1,2)]
			]
		assert result==expected_result

	def test_add_marginals_restrict_to(self):
		restrict_to = [(0,0,3,1),(3,0,1,3), (0,0,1,0), (0,0,1,1),(1,0,0,0)]
		max_k = (2,2,2,2)
		result = mut.add_marginals_restrict_to(restrict_to, max_k)
		expected_result = sorted([(0,0,0,1),(0,0,1,1),(0,0,2,1),(0,0,3,1),
			(0,0,1,0),(0,0,1,2),(1,0,1,0),(1,0,1,1),(1,0,1,2),
			(2,0,1,0),(2,0,1,1),(2,0,1,2),(3,0,1,3),(1,0,0,0)])
		assert expected_result == result
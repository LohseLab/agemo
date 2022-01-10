import itertools
import collections
import copy
import numpy as np

import gf.gflib as gflib

def return_mutype_configs(max_k, include_marginals=True):
	add_to_k = 2 if include_marginals else 1
	iterable_range = (range(k+add_to_k) for k in max_k)
	return itertools.product(*iterable_range)

def make_mutype_tree_single_digit(all_mutypes, root, max_k):
	result = collections.defaultdict(list)
	for idx, mutype in enumerate(all_mutypes):
		if mutype == root:
			pass
		else:
			if any(m>max_k_m for m, max_k_m in zip(mutype, max_k)):
				root_config = tuple(m if m<=max_k_m else 0 for m, max_k_m in zip(mutype, max_k))
				result[root_config].append(mutype)
			else:
				root_config = differs_one_digit(mutype, all_mutypes[:idx])
				result[root_config].append(mutype)
	return result

def make_mutype_tree(all_mutypes, root, max_k):
	result = collections.defaultdict(list)
	for idx, mutype in enumerate(all_mutypes):
		if mutype == root:
			pass
		else:
			if any(m>max_k_m for m, max_k_m in zip(mutype, max_k)):
				root_config = tuple(m if m<=max_k_m else 0 for m, max_k_m in zip(mutype, max_k))
				result[root_config].append(mutype)
			else:
				root_config = closest_digit(mutype, all_mutypes[:idx])
				result[root_config].append(mutype)
	return result

def differs_one_digit(query, complete_list):
	#complete_set = self.generate_differs_one_set(query)
	#similar_to = next(obj for obj in complete_list[idx-1::-1] if obj in complete_set)
	similar_to = next(obj for obj in complete_list[::-1] if sum_tuple_diff(obj, query)==1)
	return similar_to

def closest_digit(query, complete_list):
	return min(complete_list[::-1], key=lambda x: tuple_distance(query,x))

def tuple_distance(a_tuple, b_tuple):
	dist = [a-b for a,b in zip(a_tuple, b_tuple)]
	if any(x<0 for x in dist):
		return np.inf
	else:
		return abs(sum(dist))

def sum_tuple_diff(tuple_a, tuple_b):
	return sum(b-a for a,b in zip(tuple_a, tuple_b))

############ dealing with marginals ######################
def list_marginal_idxs(marginal, max_k):
	marginal_idxs = np.argwhere(marginal>max_k).reshape(-1)
	shape=np.array(max_k, dtype=np.uint8) + 2
	max_k_zeros = np.zeros(shape, dtype=np.uint8)
	slicing = [v if idx not in marginal_idxs else slice(-1) for idx,v in enumerate(marginal[:])]
	max_k_zeros[slicing] = 1
	return [tuple(idx) for idx in np.argwhere(max_k_zeros)]

def add_marginals_restrict_to(restrict_to, max_k):
	marginal_np = np.array(restrict_to, dtype=np.uint8)
	marginal_mutypes_idxs = np.argwhere(np.any(marginal_np>max_k, axis=1)).reshape(-1)
	if marginal_mutypes_idxs.size>0:
		result = []
		#for mut_config_idx in marginal_mutypes_idxs:
		#	print(marginal_np[mut_config_idx])
		#	temp = list_marginal_idxs(marginal_np[mut_config_idx], max_k)
		#	result.append(temp)
		#	print(temp)
		result = [list_marginal_idxs(marginal_np[mut_config_idx], max_k) for mut_config_idx in marginal_mutypes_idxs]
		result = list(itertools.chain.from_iterable(result)) + restrict_to
		result = sorted(set(result))
	else:
		return sorted(restrict_to)
	return result

def adjust_marginals_array(array, dimension):
	new_array = copy.deepcopy(array) #why the deepcopy here?
	for j in range(dimension):
		new_array = _adjust_marginals_array(new_array, dimension, j)
	return new_array

def _adjust_marginals_array(array, dimension, j):
	idxs = np.roll(range(dimension), j)
	result = array.transpose(idxs)
	result[-1] = result[-1] - np.sum(result[:-1], axis=0)
	new_idxs=np.zeros(dimension, dtype=np.uint8)
	new_idxs[np.transpose(idxs)]=np.arange(dimension, dtype=np.uint8)
	return result.transpose(new_idxs)

################ making branchtype dict #########################

def powerset(iterable):
	""" 
	returns generator containing all possible subsets of iterable
	"""
	s=list(iterable)
	return (''.join(sorted(subelement)) for subelement in (itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))))

def make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=None, starting_index=0):
	all_branchtypes=list(gflib.flatten(sample_list))
	branches = [branchtype for branchtype in gflib.powerset(all_branchtypes) if len(branchtype)>0 and len(branchtype)<len(all_branchtypes)]	
	if mapping.startswith('label'):
		if labels:
			assert len(branches)==len(labels), "number of labels does not match number of branchtypes"
			branchtype_dict = {branchtype: idx + starting_index for idx, branchtype in enumerate(branches)}
		else:
			branchtype_dict = {branchtype: idx + starting_index for idx, branchtype in enumerate(branches)}
	elif mapping=='unrooted': #this needs to be extended to the general thing!
		if not labels:
			labels = ['m_1', 'm_2', 'm_3', 'm_4']
		assert set(all_branchtypes)=={'a', 'b'}
		branchtype_dict=dict()
		for branchtype in gflib.powerset(all_branchtypes):
			if len(branchtype)==0 or len(branchtype)==len(all_branchtypes):
				pass
			elif branchtype in ('abb', 'a'):
				branchtype_dict[branchtype] = 1 + starting_index #hetA
			elif branchtype in ('aab', 'b'):
				branchtype_dict[branchtype] = 0 + starting_index #hetB
			elif branchtype == 'ab':
				branchtype_dict[branchtype] = 2 + starting_index #hetAB
			else:
				branchtype_dict[branchtype] = 3 + starting_index #fixed difference
	else:
		ValueError("This branchtype mapping has not been implemented yet.")
	return branchtype_dict
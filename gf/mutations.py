import itertools
import collections
import copy
import math
import numpy as np
import numba

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

def make_branchtype_dict_idxs(sample_list, phased=False, rooted=False, starting_index=0):
	samples = sorted(gflib.flatten(pop for pop in sample_list if len(pop)>0))
	if phased:
		all_branchtypes = list(flatten([[''.join(p) for p in itertools.combinations(samples, i)] for i in range(1,len(samples))]))
	else:
		all_branchtypes = list(flatten([sorted(set([''.join(p) for p in itertools.combinations(samples, i)])) for i in range(1,len(samples))]))

	fold = math.ceil(len(all_branchtypes)/2)
	correction_idx = len(all_branchtypes) - 1 if rooted else 0
	branchtype_dict = dict()
	for idx in range(fold):
		branchtype_dict[all_branchtypes[idx]] = idx + starting_index
		branchtype_dict[all_branchtypes[-idx-1]] = abs(- correction_idx + idx) + starting_index
	
	return branchtype_dict

######## initial outline general branchtype implementation ################
def get_binary_branchtype_array(dim):
	all_nums = [[p for p in itertools.combinations(range(dim), i)] for i in range(1,dim)]
	t = list(flatten(all_nums))
	x = np.zeros((len(t), dim), dtype=np.uint8)
	for idx, subt in enumerate(t):
		x[idx, subt] = 1
	return x

def permuatation_repr_binary_branchtype_array(b, slices=None):
	if slices==None:
		length_slices = b.shape[-1:]
	else:
		length_slices = slices + b.shape[-1:]
	num_permutations = np.prod([math.factorial(i) for i in length_slices])
	shape = (b.shape[0], num_permutations)
	result = np.zeros(shape, dtype = np.uint64)
	for idx, subb in enumerate(b):
		for idx2, numeric_rep in enumerate(single_branchtype_equivalences(subb, slices)):
			result[idx, idx2] = numeric_rep
	return result

def mutype_equivalences(mutype, permutation_array):
	idx = np.array(mutype)
	return np.sort(permutation_array[idx==1].T, axis=-1)

def single_branchtype_equivalences(b, slices=None):
	if slices==None:
		slices = np.array([])
	b_pop = np.split(b, slices)
	for p in all_permutations(*b_pop):
		yield branchtype_to_single_number(p)

@numba.vectorize([numba.uint8(numba.uint8, numba.uint8)])
def packbits(x, y):
	return 2*x + y

def branchtype_to_single_number(b):
	return packbits.reduce(b)

def all_permutations(*b):
	for r in itertools.product(*[single_permutation(m) for m in b]):
		yield np.hstack(tuple(r))

def single_permutation(b):
	dim = b.size
	for ordering in itertools.permutations(range(dim)):
		yield b[np.array(ordering)]

def flatten(input_list):
	#exists somewhere else
	return itertools.chain.from_iterable(input_list)

def stacked_branchtype_representation(idx, binary_btype_array):
	idx = np.array(idx)
	return binary_btype_array[idx==1]

def ravel_multi_index(multi_index, shape):
	#exists somewhere else
	shape_prod = np.cumprod(shape[:0:-1])[::-1]
	return np.sum(shape_prod * multi_index[:-1]) + multi_index[-1]

"""
way to use it:
samples = ['a','b','c', 'd']
num_mutypes = 14
mutype_array = -np.ones((2,)*num_mutypes, dtype=np.int64)
repr_dict = {}
bt_array = get_binary_branchtype_array(len(samples))
compatibility_check = branchtype_compatibility(bt_array)
permutation_array = permuatation_repr_binary_branchtype_array(bt_array)
for mutype in compatibility_depth_first(compatibility_check, num_mutypes):
	eqvs = mutype_equivalences(mutype, permutation_array)
	first = tuple(eqvs[0])
	if first in repr_dict.keys():
		representative = repr_dict[first]
		mutype_array[mutype] = representative
	else:
		for numeric_rep in eqvs:
			mutype_array[mutype] = 0
			repr_dict[tuple(numeric_rep)] = ravel_multi_index(mutype, mutype_array.shape)

"""

#determining binary_branchtype_array for unphased data

def recursive_distribute(a, idx, to_distribute, boundary):
	if to_distribute==0:
		yield tuple(a)
	else:
		if idx<len(a):
			distribute_now = min(to_distribute, boundary[idx])
			for i in range(distribute_now,-1,-1):
				new_a = a[:]
				new_a[idx] = i
				yield from recursive_distribute(new_a, idx+1, to_distribute-i, boundary)

def generate_idxs(boundary):
	total_boundary = sum(boundary)
	a = [0]*len(boundary)
	for i in range(1,total_boundary):
		yield from recursive_distribute(a, 0, i, boundary)

#from idxs to binary branchtype array
def get_binary_branchtype_array_unphased(boundary):
	result = []
	for config in generate_idxs(boundary):
		temp = []
		for idx, pop in enumerate(config):
			temp+=[1]*pop+[0]*(boundary[idx]-pop)
		result.append(tuple(temp))
	return np.array(result, dtype=np.uint8)

#testing branchtype compatibility
def branchtype_compatibility(b):
	#returns all incompatible branchtypes for a given array of branchtypes b
	num_rows = b.shape[0]
	incompatible = [set() for _ in range(num_rows)]
	for idx1, idx2 in itertools.combinations(range(num_rows), 2):
		if not intersection_check(b[idx1], b[idx2]):
			incompatible[idx1].add(idx2)
			incompatible[idx2].add(idx1)
	return incompatible

def intersection_check(a, b):
	s1 = set(np.argwhere(a).reshape(-1))
	s2 = set(np.argwhere(b).reshape(-1))
	min_len = min(len(s1), len(s2))
	len_intersection = len(s1.intersection(s2))
	return len_intersection == min_len or len_intersection==0

def compatibility_depth_first(compatibility_check, size):
	root = ((0,)*size, compatibility_check[0], size-1)
	stack = [root]
	possible = []
	while stack:
		mutype, incompatibility_set, idx = stack.pop()
		if idx>-1:
			new_mutype = list(mutype)
			new_mutype[idx] = 1
			new_mutype = tuple(new_mutype)
			stack.append((mutype, incompatibility_set, idx-1))
			if idx not in incompatibility_set:
				new_incompatibility_set = incompatibility_set.union(compatibility_check[idx])
				stack.append((new_mutype, new_incompatibility_set, idx-1))
		else:
			possible.append(mutype)
			
	return possible

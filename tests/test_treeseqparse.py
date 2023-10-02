import numpy as np
import tskit
import pytest

import agemo.treeseqparse as parse

class TestParse:
	def test_simple(self):
		ts = tskit.Tree.generate_balanced(4).tree_sequence
		bt_map = 2**np.arange(1,ts.num_samples+1)
		result = parse.track_branchtypes(ts, bt_map, mode='branch')
		present = np.array([2,4,6,8,16,24])
		assert np.all(result[0, present]== 1)
(sec_bsfs)=
# bSFS


The block-wise site frequency spectrum or bSFS is a frequency spectrum of site frequency spectrum counts in short blocks of a fixed length. For each such block of sequence one can count the number of mutations along each of the $j$ possible branchtypes and represent these as vector of length $j$. When phase information is not taken into account this vector is essentially the site frequency spectrum of that block. Tallying all these vectors results in the bSFS and can be represented as a $j$-dimensional array where entry bSFS$[x_0,x_1,...,x_j]$ represents the number of blocks with $x_i$ mutations along branchtype $i$. Similar to refering to branchtypes we will refer to each of those mutation configurations as mutationtypes.

Using the generating function we can compute the probability of observing each mutationtype. In general we will compute the probabilities for all observed mutationtypes at once and tabulate them as an array of shape $ (k_1^{max}+2, ... , k_j^{max}+2) $. Here, $k_i^{max}$ represents the maximum number of mutations along branchtype $i$ for which to calculate the associated probabilities. The $k_i^{max}+1$ entry is used to bin all mutationtypes with more than $k_i^{max}$ mutations along branchtype $i$.

To calculate the bSFS under a given structured coalescent model first specify the coalescent model. Then specify all mutationtypes for which the associated probabilities need to be tabulated using the {class}`.MutationTypeCounter`. Here mutype_shape refers to the shape of the final array and is thus defined as $ (k_1^{max}+2, ... , k_j^{max}+2) $.

## example 

This example demonstrates how to build a structured coalescent model for an isolation with migration model. This is, two populations have descended from the same ancestral population some time in the past. Since that time there has been constant unidirectional migration from one daughter population to the other. In this example we have sampled 2 lineages from each population. Phase and root information is ignored. $ k_{max} $ is set to 2 for all branchtypes.

```python
import agemo

sample_configuration = [(),('a', 'a'), ('b', 'b')]
btc = agemo.BranchTypeCounter(sample_configuration)
num_branchtypes = len(btc)

mig_event = agemo.MigrationEvent(num_variables, 1, 2)
population_split_event = agemo.PopulationSplitEvent(num_variables+1, 0, 1, 2)
events = [mig_event, population_split_event]
gfObj = agemo.GfMatrixObject(btc, events)

mutype_shape = (4,)*num_branchtypes
mtc = agemo.MutationTypeCounter(btc, mutype_shape)

bsfs_eval = agemo.BSFSEvaluator(gfObj, mtc)

theta = 72/125
theta_along_branchtypes = np.full(num_branchtypes, theta, dtype=np.float64)
params = np.array([1.0, 15/13, 5/2, 21/10])
var = np.hstack([params, theta_along_branchtypes])

bsfs_array = bsfs_eval.evaluate(theta, var, time=10/3)
```

# Quick reference


{class}`.BranchTypeCounter`
: Specify all branchtypes given a sample configuration.

{class}`.MutationTypeCounter`
: Specify all mutationtypes given a set of branchtypes.

{class}`.BSFSEvaluator`
: Builds the computational graph needed to tabulate the bSFS.

{meth}`.BSFSEvaluator.evaluate`
: Returns the probabilities for all entries of the bSFS for a particular point in parameter space.
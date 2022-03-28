# Laplace tranformed coalescence time distributions

**Agemo** allows users to generate the Laplace Transform of the (joint) coalescence time distribution of a sample with under a specified structured coalescent model for a set of distinguished branch types. It is an open-source tool with a python API that implements all the work that has been done on the generating function approach as described by {cite}`Lohse2011`. In addition, **agemo** provides ways to efficiently query that distribution, by using the fact that its generating function can be represented most simply as a directed graph with all possible ancestral states of the sample as nodes.

So far, this approach has been used to compute the probabilities of the joint site frequency spectrum for blocks of a given size (block-wise SFS or [bSFS](bSFS.md)), under models of isolation and migration {cite}`Lohse2016`, bottlenecks {cite}`Bunnefeld2015` and hard sweeps {cite}`Bisschop2021`. For this reason, the current implementation is mainly geared towards efficiently getting out the [bSFS](bSFS.md).

The potential of the generating function approach is far from being fully explored. There are lots more features possible. If you would like to contribute please open an issue or start a pull-request on GitHub.

If you use **agemo** in your work please cite
```{eval-rst}
.. todo::
	Create citation
```

```{bibliography}
```

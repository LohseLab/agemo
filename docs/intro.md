# Laplace tranformed coalescence time distributions

**Agemo** allows users to generate the Laplace Transform of the (joint) coalescence time distribution of a sample with a given demographic history for a set of distinguished branch types. It is an open-source tool with a python API that implements all the work that has been done on the generating function approach as described by {cite}`Lohse2011`. In addition, **agemo** provides ways to efficiently query that distribution, by using the fact that its generating function can be represented most simply as a directed graph with all possible ancestral states of the sample as nodes.

So far, **agemo** has been used to compute the probabilities of the joint site frequency spectrum for blocks of a given size, under models of isolation and migration {cite}`Lohse2016`, bottlenecks {cite}`Bunnefeld2015` and hard sweeps {cite}`Bisschop2021`. Calculating these probabilities requires repeated differentiation of the generating function which suffers from an explosion in the number of terms when implemented naively. Using a closed-form expression for the coefficients of a series expansion of the equations associated with each edge of the graph, we can efficiently propagate these coefficients through the graph avoiding redundant operations.


If you use **agemo** in your work please cite
```{eval-rst}
.. todo::
	Create citation
```

The potential of the generating function approach is far from being fully explored. There are lots more features possible. If you would like to contribute please open an issue or start a pull-request on GitHub.


```{bibliography}
```

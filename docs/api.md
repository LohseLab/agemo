# Python API


```{eval-rst}
.. currentmodule:: agemo
```

## Reference documentation

### Branch type and mutation type information

```{eval-rst}
.. autoclass:: agemo.BranchTypeCounter
    :members:
    :special-members: __len__
```

```{eval-rst}
.. autoclass:: agemo.MutationTypeCounter
    :members:
    :special-members: __len__
```

### Events

```{eval-rst}
.. autoclass:: agemo.MigrationEvent
    :members:
```

```{eval-rst}
.. autoclass:: agemo.PopulationSplitEvent
    :members:
```


### Generating function

```{eval-rst}
.. autoclass:: agemo.GfMatrixObject
    :members:
```

### Evaluating GF

```{eval-rst}
.. autoclass:: agemo.BSFSEvaluator
    :members: evaluate, make_product_subsetdict
```
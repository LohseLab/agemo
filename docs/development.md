(sec_development)=

# Development

If you would like to add some features to `agemo`, please read the
following. If you think there is anything missing,
please open an [issue](<http://github.com/lohselab/agemo/issues>) or
[pull request](<http://github.com/lohselab/agemo/pulls>) on GitHub!

(sec_development_quickstart)=

## Quickstart

- Make a fork of the agemo repo on [GitHub](<http://github.com/lohselab/agemo>)
- Clone your fork into a local directory. 

  ```
  $ git clone git@github.com:LohseLab/agemo.git
  ```

- Install the `requirements/dev.txt`.
- Run the tests to ensure everything has worked: `python3 -m pytest`. These should
  all pass.
- Install the pre-commit checks: `pre-commit install`
- Make your changes in a local branch. On each commit a [pre-commit hook](<https://pre-commit.com/>)  will run
  checks for code style and common problems.
  Sometimes these will report "files were modified by this hook" `git add`
  and `git commit --amend` will update the commit with the automatically modified
  version.
  The modifications made are for consistency, code readability and designed to
  minimise merge conflicts. They are guaranteed not to modify the functionality of the
  code. To run the checks without committing use `pre-commit run`. To bypass
  the checks (to save or get feedback on work-in-progress) use `git commit
  --no-verify`
- When ready open a pull request on GitHub. Please make sure that the tests pass before
  you open the PR, unless you want to ask for help with a failing test.

(sec_development_tests)=
## Tests
All tests are in the `tests` directory, and run using
[pytest](<https://docs.pytest.org/en/stable/>).

Specifc tests can be run using
`pytest tests/test_evaluate.py::TestTaylorSinglePop`.

Some `agemo` tests make use of [sympy](<https://docs.sympy.org/latest/index.html>) to verify its
functionality using a computer algebra system that allows us to deal with symbolic maths.

(sec_development_documentation)=
## Documentation

### Documentation quickstart

- Install the `requirements/docs.txt`
- To build the documentation locally, go to the `docs` directory and run `make`. 

### JupyterBook

Documentation for agemo is built using [Jupyter Book](https://jupyterbook.org),
which allows us to mix API documentation generated automatically using
[Sphinx](https://www.sphinx-doc.org) with code examples evaluated in a
local [Jupyter](https://jupyter.org) kernel. This is a very powerful
system that allows us to generate beautiful and useful documentation,
but it is quite new and has some quirks and gotchas.

#### reStructuredText

All of the documentation for previous versions of agemo was written
using the [reStructuredText](<https://docutils.sourceforge.io/rst.html>) format
(rST) which is the default for Python documentation. Because of this, all of
the API docstrings are written using rST. Converting these docstrings to Markdown
would be a lot of work (and support from upstream tools for Markdown
dosctrings is patchy), and so we need to use rST for this
purpose for the forseeable future.

Some of the directives we use are only available in rST, and so these
must be enclosed in ``eval-rst`` blocks like so:

````md
```{eval-rst}
.. autoclass:: agemo.GfMatrixObject
```
````

#### Markdown

Everything **besides** API docstrings is written using
[MyST Markdown](https://jupyterbook.org/content/myst.html). This is a
superset of [common Markdown](https://commonmark.org) which
enables executable Jupyter content to be included in the documentation.
In particular, JupyterBook and MyST are built on top of
[Sphinx](https://www.sphinx-doc.org) which allows us to do lots
of cross-referencing.

Some useful links:

- The [MyST cheat sheet](https://jupyterbook.org/reference/cheatsheet.html)
  is a great resource.
- The "Write Book Content" part of the [Jupyter Book](https://jupyterbook.org/)
  documentation has lots of helpful examples and links.
- The [MyST Syntax Guide](https://myst-parser.readthedocs.io/en/latest/using/syntax.html)
  is a good reference for the full syntax
- Sphinx
  [directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html).
  Some of these will work with Jupyter Book, some won't. There's currently no
  comprehensive list of those that do. However, we tend to only use a small subset
  of the available directives, and you can usually get by following existing
  local examples.
- The [types of source files](https://jupyterbook.org/file-types/index.html)
  section in the Jupyter Book documentation is useful reference for mixing
  and matching Markdown and rST like we're doing.

## Disclaimer

The development section was copied from the wonderful [msprime docs](<https://tskit.dev/msprime/docs/stable/development.html>).
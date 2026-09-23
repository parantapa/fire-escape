# Development notes

Notes for working on Fire-ESCAPE itself.
For using it, start at the [README](../README.md)
and the [documentation index](../README.md#documentation).

## The shape of the project

Fire-ESCAPE is a compiler.
It reads a model written in FFSL
and writes a C++ project that builds into a simulator.
Nothing in this repository simulates anything.
The Python here runs once, at compile time.
Everything that costs time at run time
lives in the Jinja template that the compiler fills in.

That split explains most of the layout below.
A change to how a fire spreads is a change to the template.
A change to what a model is allowed to say
is a change to the grammar, the AST and the type checker together.

## Map of the source

| Path | Holds |
| --- | --- |
| `src/fire_escape/cli.py` | The `ffsc` entry point, and the top level error handling. |
| `src/fire_escape/grammar.lark` | The FFSL grammar, as a LALR grammar with an indenter. |
| `src/fire_escape/parser.py` | Parse tree to AST, name resolution, and the post-parse passes. |
| `src/fire_escape/ast_nodes.py` | Every node type, as a pydantic model. |
| `src/fire_escape/type_check.py` | The type lattice, and the pass that annotates expressions. |
| `src/fire_escape/builtins.py` | The names a model gets without declaring them. |
| `src/fire_escape/error.py` | The exception hierarchy, and source positions. |
| `src/fire_escape/codegen_openmp_cpu.py` | The OpenMP CPU backend. |
| `src/fire_escape/templates/__init__.py` | Loading many named templates out of one `.jinja` file. |
| `src/fire_escape/templates/openmp-cpu.jinja` | The generated `simulator.cpp`, `CMakeLists.txt` and `conanfile.py`. |
| `examples/example1.ffsl` | The model the end-to-end test compiles. |
| `tests/run-example1.py` | The end-to-end test: compile, build, run, check. |
| `tests/check_output.py` | The output checks, also usable on its own. |
| `docs/` | The user documentation, and these notes. |

## Build and run

```
pip install -e .[dev]
```

`pyproject.toml` requires Python 3.13 or newer.

That puts `ffsc` on your path
and adds `pytest`, `black`, `pyright`, `h5py` and `polars`.

To compile a model:

```
ffsc compile -i examples/example1.ffsl -o build/sim
```

Building the result needs Conan, CMake and a C++23 compiler,
which are not Python dependencies and are not installed by the line above.
[How to compile and run a model](how-to-guides/compile-and-run-a-model.md)
covers that side.

## Checks

Run black first and pyright second,
so pyright checks the code as it will be committed.

```
black --target-version py313 src tests
pyright src tests
```

Neither tool has a configuration section in `pyproject.toml`,
so both run with their defaults.
Pass `--target-version py313` to black.
Without it, black infers a newer target
and warns that Python 3.13 cannot run its safety check.

## Tests

The end-to-end test is a script rather than a pytest module.
It needs a C++ toolchain and a dataset.
A developer without either can then skip it
instead of watching it fail.

```
tests/run-example1.py \
    -m examples/example1.ffsl \
    -d <directory holding t1-tiles.h5, t1-seeds.h5, t1-tick.parquet> \
    -w build/test-example1
```

Pass `--skip-build` to reuse the simulator an earlier run built.
Anything after `--` goes to the simulator,
so a config option can be exercised without editing the model.

`tests/check_output.py` runs on its own against any output file:

```
python tests/check_output.py verify \
    --tile-file tiles.h5 --seed-file seeds.h5 \
    --tick-file ticks.parquet --output-file output.h5
```

The checks are invariants rather than golden values.
Among other things, they assert these three.

* No tile moves backwards through the three states.
* The tallies match the saved grids.
* No tick receives more embers than it generated.

They report how far the fire spread rather than assert it.
The reason is a comment in `verify_series` in `tests/check_output.py`.

## Tools

| Tool | Used for | Where |
| --- | --- | --- |
| lark | The parser, LALR with a custom `Indenter`. | `grammar.lark`, `parser.py` |
| pydantic | The AST node types. | `ast_nodes.py` |
| networkx | The type conversion lattice. | `type_check.py` |
| jinja2 | The generated C++. | `templates/`, `codegen_openmp_cpu.py` |
| click | The `ffsc` command line, and the end-to-end test script. | `cli.py`, `codegen_openmp_cpu.py`, `tests/run-example1.py` |
| json5 | The template headers inside a `.jinja` file. | `templates/__init__.py` |
| h5py | Reading the tile, seed and output files in the checks. | `tests/check_output.py` |
| polars | Reading the tick file in the checks. | `tests/check_output.py` |
| black | Formatting. See Checks. | all Python files |
| pyright | Type checking. See Checks. | all Python files |

The generated project pulls its own C++ dependencies through Conan:
fmt, argparse, mdspan, Random123, Arrow and HDF5.
The conanfile template also pins the boost that Arrow pulls in.
The reason is a comment at the pin in `templates/openmp-cpu.jinja`.

## Design decisions

These cross more than one file.
A decision local to one place is a comment there instead.

### Division is always true division

`/` in FFSL follows Python, not C.
Two integer operands still produce a `float`.

This is settled in two files that have to agree.
`TypeEnv.check_binary` in `type_check.py` returns `float` for `/`,
whatever the operands were.
`codegen_expr` in `codegen_openmp_cpu.py`
inserts a `float(...)` conversion on the left operand,
when that operand is an integral C++ type.

Neither half works without the other.
Without the conversion, the type checker promises a float
that C++ does not deliver.
Without the type checker, the conversion leaves the declared types wrong.

The cost is that integer division is unreachable from the surface language,
and that a function returning `int` cannot return a division directly.
`check_update` rejects `n /= 2` on an integral `n` for the same reason.
The alternative, a separate `//` operator, was not taken,
which keeps the language smaller
at the price of that one missing operation.

### Positions are attributed by the innermost frame that knows one

A `CodeError` is usually raised deep in a check
that does not know which line it is looking at.
`node_error_attributer` in `error.py` wraps the tree-walking passes.
It fills in the position of the node it visits,
and only when the error does not already carry one.

The effect is that the innermost frame with a position wins,
so an error reports the subexpression at fault
rather than the declaration containing it.
A new tree-walking pass needs the decorator,
or its errors will surface with no position at all.

`check_type` in `type_check.py` is the one exception.
It repeats the decorator's logic in its own `except` clauses
instead of taking the decorator.

### A compiler fault and a model fault are different exceptions

`CodeError` means the model is wrong,
and it prints as one located line with no traceback.
`CompilerError` means Fire-ESCAPE is wrong,
and it keeps its traceback so it can be reported.
`main` in `cli.py` is where that split becomes visible,
and the tree-walking passes preserve it
by re-raising `CompilerError` untouched
while rewrapping anything else.

The practical rule when adding a check:
a `raise` reachable from a model a user can write is a `CodeError`.
A `raise` reachable only from a tree the grammar cannot produce
is a `CompilerError`.

### The AST carries three kinds of edge, and only one is `children`

Every node has `children`, and the generic walks recurse through it.
Nodes also carry edges that are not children:
`Ref.scope` and `Func.scope` point sideways into a `ChainMap`,
and `ReturnStmt.func` points back up to its function.
`Func.lvars` and `Func.return_stmts` point down,
but stay out of `children` too,
because a walk already reaches those nodes through the function's block.

Those are kept out of `children` deliberately.
A walk over `children` terminates.
A walk that follows `ReturnStmt.func` runs forever.

A new field that points at another node needs two decisions.
First, decide which of the two kinds of edge it is.
Then set `repr=False` on it where it points anywhere but down.
Without that, the pydantic repr in an error message
prints the whole tree.

### The post-parse passes run in a fixed order

`parse` in `parser.py` walks the tree four times
before the type checker sees it,
and then runs three checks over the declarations.
The order is load bearing.

`build_scope` has to precede `populate_tile_objects`.
The first creates the tile bindings that the second fills.
The work is split in two because a model can declare `tile-data`
after the `fire-model` block that refers to it.

`link_return_statements` has to precede `check_type`,
because the return type check reads `ReturnStmt.func`.

A new pass goes wherever those dependencies allow.

### Config values of type `float` are stored as `double`

The argparse library the generated simulator uses
parses a `double` but not a `float`.
So a `config` declared `float` is stored as a `double`
and narrowed at each use.
`cpp_type_config` in `codegen_openmp_cpu.py` is the widening,
and `codegen_expr` inserts the `static_cast` back down
whenever the two disagree.

This is why a run echoes `base_ember_rate = 0.699999988079071`
for a default written as `0.7`.
The same option passed on the command line echoes `0.9`.
The default made the trip through `float`, and the flag did not.

### Generated statements carry the model's line numbers

Every statement template in `templates/openmp-cpu.jinja`
opens with `#line <n> "<model file>"`,
so a C++ compiler error inside generated model code
points at the line of the model that produced it.
Each such block closes with `#endline`, which is not a directive.
`codegen` in `codegen_openmp_cpu.py` replaces every `#endline`
with a `#line` back into `simulator.cpp`,
numbered for the line that follows it.

The two files have to agree on the marker.
A new statement template needs both lines,
or the code after it is attributed to the model.

### Two error classes shadow builtins

`error.py` defines `ReferenceError` and `TypeError`,
which shadow the Python builtins of the same name.
Both are in `__all__`,
so `ast_nodes.py`, `parser.py` and `type_check.py`,
which all do `from .error import *`,
get the compiler's versions.

The effect is that `raise TypeError(...)` in those files raises a `CodeError`,
which prints as a located model error.
Catching the Python builtin in those files needs `builtins.TypeError`.
Renaming the classes would remove the trap,
at the cost of touching every raise site.

## Known rough edges

* `chunk_size_x` and `chunk_size_y` do nothing yet.
  The comment in `populate_source_opts` in `parser.py` has the detail.
* `MyIndenter` mishandles mixed tabs and spaces.
  The comment at `tab_len` in `parser.py` has the detail.
* `build_ast` in `parser.py` has no return annotation, on purpose.
  Its truthful type is `AstNode | str`.
  Writing that down makes pyright check every node construction
  in the function, which then reports more than a hundred errors.
  The Lark children lists are heterogeneous,
  and they go straight into strictly typed pydantic fields.
  The function is recursive, so pyright otherwise infers Unknown
  for the recursive calls that fill the children lists.
  The whole file then type checks clean on a false premise.
  Fixing this means narrowing the children lists at each `case`,
  not deleting the annotation again.
* A syntax error in a model is not a `CodeError`.
  Lark raises it from `parse` as an `UnexpectedInput` or a `DedentError`,
  so `main` in `cli.py` lets it through with a traceback.

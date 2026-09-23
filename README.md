# Fire-ESCAPE

Fire-ESCAPE is a framework for creating high-performance wildfire spread simulators.

A wildfire spread model is written
in the Forest Fire Simulation Language (FFSL).
The `ffsc` compiler turns that model into a parallel shared memory
C++ simulator, which uses OpenMP for multi-threading.

## Installation

Requires Python 3.13 or later.

```
pip install .
```

This installs the `ffsc` command line tool.
Building a generated simulator also needs Conan, CMake and a C++23 compiler.

## Usage

Compile a model to a C++ project.

```
ffsc compile -i examples/example1.ffsl -o build/sim
```

The command writes three files into the output directory.

```
build/sim/simulator.cpp    the generated simulator
build/sim/CMakeLists.txt   the CMake build definition
build/sim/conanfile.py     the Conan recipe listing the C++ dependencies
```

Building that project produces the simulator binary.
The binary reads its grid from HDF5 files
and its per-tick data from a Parquet file,
and writes an HDF5 output file.
[Your first simulation](docs/tutorials/your-first-simulation.md)
walks through the whole path from this command to a finished run.

## Documentation

| Document | Contents |
| --- | --- |
| [Your first simulation](docs/tutorials/your-first-simulation.md) | Compile the example model, build it, and watch a fire cross a grid. |
| [How to compile and run a model](docs/how-to-guides/compile-and-run-a-model.md) | The steps from an FFSL file to an output file, for a model of your own. |
| [The FFSL language](docs/reference/ffsl-language.md) | The six top-level declarations, the type set, the statements and the builtins. |
| [The simulator command line](docs/reference/simulator-command-line.md) | Every flag a generated simulator takes, including the `--config-*` flags. |
| [Input and output files](docs/reference/input-and-output-files.md) | The layout of the tile, seed, tick and output files. |
| [The simulation model](docs/explanation/simulation-model.md) | The eight phases of a tick, and the design decisions behind them. |
| [Reproducibility](docs/explanation/reproducibility.md) | What the seed fixes, and why several threads do not repeat. |

## Contributing

| Document | Contents |
| --- | --- |
| [Development notes](docs/developer-notes.md) | The source map, the build, the tests, and the design decisions that cross files. |

## License

MIT. See `LICENSE`.

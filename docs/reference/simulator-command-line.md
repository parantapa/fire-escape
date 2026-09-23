# The simulator command line

The simulator that `ffsc compile` produces
takes its grid size, its input files and its output file on the command line.
Some flags depend on the FFSL source the simulator was compiled from.

## Required flags

| Flag | Short | Value |
| --- | --- | --- |
| `--num-ticks` | `-n` | Number of ticks to simulate. |
| `--num-rows` | `-r` | Rows in the tile grid. |
| `--num-cols` | `-c` | Columns in the tile grid. |
| `--tile-file` | `-t` | Tile data file, in HDF5. |
| `--seed-file` | `-s` | Seed data file, in HDF5. |
| `--tick-file` | `-w` | Tick data file, in Parquet. |
| `--output-file` | `-o` | Output file, in HDF5. |

The file formats are given in
[Input and output files](input-and-output-files.md).

## Optional flags

| Flag | Short | Default | Value |
| --- | --- | --- | --- |
| `--seed` | | `-1` | RNG seed. A negative value draws a fresh seed from the system. |
| `--help` | `-h` | | Show the usage message and exit. |
| `--version` | `-v` | | Print version information and exit. |

## Config flags

Every `config` declared in the FFSL source
also becomes a command line flag,
named `--config-` followed by the variable name,
with underscores replaced by hyphens.
A model that declares this

```
config base_ember_rate: float = 0.7
```

accepts `--config-base-ember-rate 0.9`,
which overrides the default without recompiling the model.

## Environment variables

The simulator uses OpenMP,
so the thread count is taken from the `OMP_NUM_THREADS` environment variable
rather than from a flag.

## Startup output

The simulator echoes its resolved settings before the first tick,
including the effective seed.
The seed is also saved as the `seed` attribute
of the `runstats` group in the output file.
A single-threaded run therefore repeats from its own output.

```
NUM_TICKS = 20
NUM_ROWS = 64
NUM_COLS = 64
seed_file_name = data/seeds.h5
tick_file_name = data/ticks.parquet
tile_file_name = data/tiles.h5
output_file_name = data/output.h5
base_ember_rate = 0.699999988079071
base_ignition_rate = 0.5
seed = 42
num_threads = 4
total_alloc_gb = 0.023822288
```

Reproducibility across runs is discussed in
[Reproducibility](../explanation/reproducibility.md).

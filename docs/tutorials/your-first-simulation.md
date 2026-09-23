# Your first simulation

In this lesson we take the example model that ships with Fire-ESCAPE
and turn it into a simulator.
Then we watch a fire spread from one burning tile
across a 64 by 64 grid.

We will end with a table of the fire, tick by tick.

Everything below runs from the root of the repository.

## Before we start

We need `ffsc`, Conan, CMake, a C++ compiler,
and the Python packages `h5py`, `numpy` and `polars`.
Installing Fire-ESCAPE with its development extras gives us all of the Python side.

```
pip install -e '.[dev]'
```

Let us make a place to work.

```
mkdir -p tutorial/data
```

## 1. Look at the model we are going to compile

Open `examples/example1.ffsl`.
We are not going to change it, so a glance is enough.

Notice the `tile-data` block near the top.

```
tile-data:
    burn_time: float
    fuel: float
    moisture: float

    pos: position
    state: fire_state seed save
```

The three `float` variables carry no annotation,
so we will have to supply them.
`pos` is a `position`, which the simulator fills in itself.
`state` is marked `seed`, so it comes from a separate file.
It is the one that says where the fire starts.
It is also marked `save`, so it comes back to us in the output.

Notice too the `tick-data` block, which asks for `wind_speed` and `wind_direction`,
keyed by a `tick` column.
We will supply those as well.

## 2. Build the input files

The repository ships a model but no data,
so we make our own.
Put this in `tutorial/make_inputs.py`.

```python
"""Build a small input dataset for the tutorial."""

import h5py
import numpy as np
import polars as pl

NUM_ROWS, NUM_COLS, NUM_TICKS = 64, 64, 20

rng = np.random.default_rng(0)

with h5py.File("tutorial/data/tiles.h5", "w") as f:
    f["fuel"] = rng.uniform(0.5, 1.0, (NUM_ROWS, NUM_COLS))
    f["moisture"] = rng.uniform(0.0, 0.5, (NUM_ROWS, NUM_COLS))
    f["burn_time"] = rng.uniform(0.2, 0.6, (NUM_ROWS, NUM_COLS))

state = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.int8)
state[32, 32] = 1
with h5py.File("tutorial/data/seeds.h5", "w") as f:
    f["state"] = state

pl.DataFrame(
    {
        "tick": np.arange(NUM_TICKS, dtype=np.int64),
        "wind_speed": np.full(NUM_TICKS, 2.0),
        "wind_direction": np.full(NUM_TICKS, 0.0),
    }
).write_parquet("tutorial/data/ticks.parquet")

print(f"{NUM_ROWS} x {NUM_COLS} grid, {NUM_TICKS} ticks, 1 burning tile")
```

Notice that each dataset is named after a variable in the model.
The one burning tile is `state[32, 32] = 1`,
right in the middle of the grid.
That single `1` is the whole fire we are about to start.

Run it.

```
python3 tutorial/make_inputs.py
```

```
64 x 64 grid, 20 ticks, 1 burning tile
```

## 3. Compile the model

```
ffsc compile -i examples/example1.ffsl -o tutorial/sim
```

The command prints nothing.
That silence is success.
Let us see what it produced.

```
ls tutorial/sim
```

```
CMakeLists.txt
conanfile.py
simulator.cpp
```

`ffsc` did not run anything.
It wrote out a complete C++ project,
and building that project is our next job.

## 4. Build the simulator

This step downloads and compiles the C++ dependencies the first time,
so it will take a few minutes.
Later runs reuse what it caches.

```
cd tutorial/sim
conan build . --build=missing -s compiler.cppstd=gnu23
```

The last two lines tell us it worked.

```
[100%] Linking CXX executable simulator
[100%] Built target simulator
```

Let us go back to the repository root.

```
cd ../..
```

The binary is at `tutorial/sim/build/Release/simulator`.

## 5. Run the fire

We pass the grid size and the tick count ourselves.
They have to match the files we made in step 2.
We also pin the seed and the thread count.
Our numbers and the numbers below are then the same.

```
OMP_NUM_THREADS=1 tutorial/sim/build/Release/simulator \
    --num-ticks 20 \
    --num-rows 64 \
    --num-cols 64 \
    --tile-file tutorial/data/tiles.h5 \
    --seed-file tutorial/data/seeds.h5 \
    --tick-file tutorial/data/ticks.parquet \
    --output-file tutorial/data/output.h5 \
    --seed 42
```

The output starts with the settings, then counts the ticks.

```
NUM_TICKS = 20
NUM_ROWS = 64
NUM_COLS = 64
seed_file_name = tutorial/data/seeds.h5
tick_file_name = tutorial/data/ticks.parquet
tile_file_name = tutorial/data/tiles.h5
output_file_name = tutorial/data/output.h5
base_ember_rate = 0.699999988079071
base_ignition_rate = 0.5
seed = 42
num_threads = 1
total_alloc_gb = 0.023813648
tick = 0
tick = 1
```

Notice `base_ember_rate` and `base_ignition_rate` in that list.
Those are the two `config` values declared in the model,
and the simulator reports the value it is actually using.
Remember them, because we come back to one of them in step 7.

The last two lines close the run.

```
log_prob = -2184404.434282512
runtime_s = 0.073760048
```

`log_prob` is the log probability of everything the simulator sampled.
It is the quickest way to tell two runs apart.

## 6. Look at the fire

The output file holds a per-tick count of each state.
Put this in `tutorial/show_result.py`.

```python
"""Print the per tick state counts of a run."""

import h5py

with h5py.File("tutorial/data/output.h5") as f:
    unburned = f["num_unburned"][:]
    burning = f["num_burning"][:]
    burnt_out = f["num_burnt_out"][:]
    seed = f["runstats"].attrs["seed"]

print(f"seed = {seed}")
print(f"{'tick':>4} {'unburned':>9} {'burning':>8} {'burnt out':>10}")
for tick in range(len(burning)):
    print(f"{tick:>4} {unburned[tick]:>9} {burning[tick]:>8} {burnt_out[tick]:>10}")
```

```
python3 tutorial/show_result.py
```

```
seed = 42
tick  unburned  burning  burnt out
   0      4091        5          0
   1      4083       13          0
   2      4065       31          0
   3      4038       58          0
   4      4001       95          0
   5      3956      137          3
   6      3894      198          4
   7      3826      264          6
   8      3754      328         14
   9      3663      402         31
  10      3574      468         54
  11      3470      542         84
  12      3347      639        110
  13      3231      707        158
  14      3097      768        231
  15      2941      865        290
  16      2777      953        366
  17      2598     1041        457
  18      2432     1120        544
  19      2290     1143        663
```

There is our fire.

Notice three things in that table.

Tick 0 already shows five burning tiles, not the one we seeded.
The counts are taken at the end of a tick.
By then the tile we lit threw its first embers, and they landed.

The `burnt out` column stays at zero until tick 5.
Tiles burn for a while before they stop.
`burn-time` in the model is what decides how long.

The `seed` line comes from the output file, not from our command.
Every run records the seed it used.
[Reproducibility](../explanation/reproducibility.md) shows
how that lets a single-threaded run be repeated from its own output.

## 7. Change a parameter and run it again

We do not need to recompile to change a `config`.
Let us make the fire throw more embers.

```
OMP_NUM_THREADS=1 tutorial/sim/build/Release/simulator \
    --num-ticks 20 \
    --num-rows 64 \
    --num-cols 64 \
    --tile-file tutorial/data/tiles.h5 \
    --seed-file tutorial/data/seeds.h5 \
    --tick-file tutorial/data/ticks.parquet \
    --output-file tutorial/data/output.h5 \
    --seed 42 \
    --config-base-ember-rate 0.9
```

Notice that the startup output now reports `base_ember_rate = 0.9`,
and that `log_prob` has moved from `-2184404.434282512`
to `-2916956.638894377`.

Run `python3 tutorial/show_result.py` again.
Notice that at tick 19 there are now 1282 burning tiles,
where the first run had 1143.

Notice that the early ticks did not grow with it.
Tick 0 shows four burning tiles where the first run showed five.
A parameter change moves which random numbers get drawn where,
so these are two different fires,
and only the trend across them is the answer.
[Reproducibility](../explanation/reproducibility.md) explains why.

Run step 5 once more, without `--config-base-ember-rate`,
and the original table comes back exactly.
That is worth doing.
A run we can return to is what makes the next change readable.

## What we did

We compiled a model into a C++ project
and built it into a simulator.
We gave it a grid with one burning tile.
Then we watched a fire cross that grid twice,
under two different parameters.

From here:

* [How to compile and run a model](../how-to-guides/compile-and-run-a-model.md)
  covers the same ground for a model and a dataset of your own.
* [The FFSL language](../reference/ffsl-language.md)
  describes everything `examples/example1.ffsl` uses.
* [The simulation model](../explanation/simulation-model.md)
  explains what happened between tick 0 and tick 19.

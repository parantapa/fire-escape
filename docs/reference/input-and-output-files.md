# Input and output files

A generated simulator reads three input files
and writes one output file.
The variables that make up each file are declared in the FFSL source.
The declarations are described in
[The FFSL language](ffsl-language.md).

## Tile file (HDF5)

One two-dimensional dataset per non-seeded tile variable,
shaped `(num_rows, num_cols)`, named after the variable.

Given the `tile-data` block below,
the tile file holds `burn_time`, `fuel` and `moisture`.

```
tile-data:
    burn_time: float
    fuel: float
    moisture: float

    pos: position
    state: fire_state seed save
```

The variable of type `position` is synthetic
and is never read from the tile file.

## Seed file (HDF5)

The same layout as the tile file,
but holding the tile variables marked `seed`.
This is the initial condition, including which tiles start out burning.

## Tick file (Parquet)

One row per tick,
one column per tick variable,
with the column of the variable marked `key` giving the tick index.
Rows outside `[0, num_ticks)` are ignored.

## Output file (HDF5)

The output file holds three kinds of data.

### Per-tile series

One dataset per tile variable marked `save`,
shaped `(num_ticks, num_rows, num_cols)`,
written incrementally as the run proceeds.

### Per-tick summary series

One dataset each, of length `num_ticks`.

| Dataset | Contents |
| --- | --- |
| `num_unburned` | Tiles in the `Unburned` state at the end of the tick. |
| `num_burning` | Tiles in the `Burning` state at the end of the tick. |
| `num_burnt_out` | Tiles in the `BurntOut` state at the end of the tick. |
| `num_ember_gen_sum` | Embers generated during the tick. |
| `num_ember_recv_sum` | Embers received during the tick. |
| `num_flame_gen_sum` | Flame generated during the tick. |
| `num_flame_recv_sum` | Flame received during the tick. |

### The `runstats` group

Attributes on the group hold the run totals.

| Attribute | Contents |
| --- | --- |
| `seed` | The RNG seed the run actually used. |
| `log_prob` | The log probability of the run. |
| `total_alloc_bytes` | Total allocation size of the run. |
| `read_input_duration` | Wall-clock seconds spent reading the input files. |
| `init_duration` | Wall-clock seconds spent on initialization, including reading the input files. |
| `main_duration` | Wall-clock seconds spent in the main tick loop. |
| `write_output_duration` | Wall-clock seconds spent writing the per-tile series. |
| `prog_duration` | Wall-clock seconds for the whole program, up to writing the `runstats` group. |

Datasets inside the group hold a per-thread, per-tick duration
for each simulation phase:
`reset_for_tick_duration`,
`create_embers_duration`,
`create_flames_duration`,
`compute_jump_dist_duration`,
`assign_embers_duration`,
`assign_flames_duration`,
`do_ignition_duration`,
`do_transition_duration`,
and `count_states_duration`.

The phases these durations belong to are described in
[The simulation model](../explanation/simulation-model.md).

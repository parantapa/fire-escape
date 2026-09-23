# The simulation model

A wildfire in Fire-ESCAPE spreads by two mechanisms at once.
Embers travel far and land at random.
Flame spreads only to the eight adjacent tiles.
Both mechanisms feed the same ignition test,
and the eight clauses of an FFSL
[`fire-model` block](../reference/ffsl-language.md#fire-model)
are the places where a model author controls them.

Understanding the order of the phases
explains why a model behaves the way it does,
because each phase reads the state the previous phase wrote.

## The eight phases of a tick

Each tick first runs a reset phase, `reset_for_tick`,
which zeroes the per-tick tile counters and the jump tables.
It then executes the following phases,
all of them parallel loops over the grid.

1. **Create embers.**
   Every `Burning` tile samples a count of embers
   from the Poisson distribution given by `create-embers`,
   clamped to `max_ember_count`.
2. **Create flames.**
   Every `Burning` tile emits the flame intensity given by `create-flames`.
3. **Compute jump distribution.**
   For each ember source, `ember-jump-likelihood` is evaluated
   over the `(2 * max_jump_x + 1) x (2 * max_jump_y + 1)` neighborhood,
   and the weights are compiled into an alias table.
4. **Assign embers.**
   Each ember samples a landing tile from the alias table
   and survives the flight with probability `1 - ember-death-prob`.
5. **Assign flames.**
   Flame is distributed to the eight adjacent tiles,
   scaled by `flame-spread-weight`.
6. **Ignition.**
   An `Unburned` tile that received embers or flame ignites with probability
   `1 - (1 - p_ember)^n_embers * (1 - p_flame)^n_flame`.
   On ignition its burn time is drawn from `burn-time`.
7. **Transition.**
   Tiles whose burn time elapsed become `BurntOut`.
8. **Count states.**
   Per tick tallies are reduced across threads.

## Why the weights become an alias table

`ember-jump-likelihood` returns an unnormalized weight
rather than a probability.
A model author is then free to write the shape of the distribution,
without regard for what it sums to.
The cost of that freedom falls on the simulator.
It has to normalize an arbitrary discrete distribution and then sample it,
once per ember, for every burning tile, on every tick.

A naive approach walks the cumulative weights,
which costs time proportional to the size of the neighborhood.
With `max_jump_x` and `max_jump_y` set to 8,
that neighborhood holds 289 entries.
The alias method pays a setup cost proportional to the neighborhood,
once, in phase 3.
It then draws each sample in constant time in phase 4.
The trade pays off whenever a tile throws more than a few embers,
which is the case the language is built for.

## Why the ember count is clamped

`create-embers` returns a Poisson distribution,
which has no upper bound.
Phase 4 draws a landing tile for each ember in turn,
so an unbounded count means unbounded work for a single tile.
`max_ember_count` is the bound on that count.
It is a compile-time [`option`](../reference/ffsl-language.md#option)
rather than a run-time [`config`](../reference/ffsl-language.md#config),
and the generated simulator carries it as a constant.
No storage is sized by it.

The clamp is a real approximation, not a formality.
A model whose rate sits close to `max_ember_count`
loses the tail of its distribution,
and the loss grows silently as the rate rises.

## Why ignition combines the two mechanisms multiplicatively

The ignition rule treats every received ember and every unit of received flame
as an independent trial.
The probability that none of them ignites the tile
is the product of the individual failure probabilities.
The probability that at least one does is one minus that product.

The consequence is that embers and flame do not compete.
A tile that receives a little of each
is more likely to ignite than a tile that receives either alone.
A model author who wants one mechanism to dominate
has to say so through the rates,
because the combination rule itself has no preference.

## Where the phases show up in the output

Each phase contributes a per-thread, per-tick duration dataset
to the `runstats` group of the output file.
That is what makes a slow model diagnosable phase by phase.
The datasets are listed in
[Input and output files](../reference/input-and-output-files.md).

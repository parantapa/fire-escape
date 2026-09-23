# Reproducibility

A wildfire simulation is a Monte Carlo computation.
Two runs of the same model over the same inputs
produce different fires unless the random number stream is pinned down.
Fire-ESCAPE pins it down with a single seed.
The guarantee that seed buys is narrower than it first appears.
It holds on one thread.
It does not hold on several, even at the same thread count.

## What the seed fixes

The `--seed` flag of the
[simulator command line](../reference/simulator-command-line.md)
sets the RNG seed.
A negative value, which is the default, draws a fresh seed from the system.
The effective seed is echoed at startup,
and it is saved as the `seed` attribute of the `runstats` group
in the [output file](../reference/input-and-output-files.md#output-file-hdf5).
A run can therefore be repeated from its own output,
without anyone having recorded the seed at the time.

Two single-threaded runs with the same seed
produce identical output, apart from the timings in the `runstats` group,
and an identical `log_prob`.
The `log_prob` is the cheaper of the two to compare,
because it reduces the whole run to one number.

The same seed does not make two runs with different parameters comparable
tile by tile.
A parameter change moves which random numbers get drawn where,
so the second run is a different fire,
not the same fire with a dial turned.
Only the trend across such runs is the answer.

## Why several threads break the repeat

The generator is Philox, from Random123,
and each thread builds its engine from the pair `(seed, thread number)`.
Every thread therefore draws from a stream of its own.
A draw belongs to the thread that made it,
rather than to the tile it was made for.

Every [phase of a tick](simulation-model.md#the-eight-phases-of-a-tick)
after the reset is an OpenMP loop over the grid
under `schedule(guided)`.
A guided schedule hands the next chunk of tiles
to whichever thread finishes first.
The timing of the run therefore decides
which thread receives which tile.
Two runs at the same thread count, from the same seed,
can hand tile `(32, 32)` to different threads.
The tile then draws a different number.

The effect is easy to see.
Five runs of the example model over the same inputs, all with `--seed 42`,
give these values.

| Threads | `log_prob` |
| --- | --- |
| 1 | -2184404.434282512 |
| 1 | -2184404.434282512 |
| 4 | -2309581.42674017 |
| 4 | -2424208.1878262963 |
| 8 | -2182903.17992609 |

The two single-threaded runs agree exactly.
The two four-thread runs do not agree with each other.

The fires that result are statistically equivalent.
They are not identical, and `log_prob` will not match.

## The choice behind this

An alternative design keys the random stream to the tile
rather than to the thread.
Philox is a counter-based generator, so it supports exactly that.
The counter can be built from the tile index and the tick,
instead of from a per-thread sequence.
Output is then identical at every thread count,
and under any schedule.

The cost is that each draw becomes a keyed hash of a fresh counter,
rather than an advance of a stream the thread already holds.
That cost is paid on every ember of every burning tile of every tick.
A guided schedule also loses part of its point.
It is there to balance the uneven work
that a spreading fire creates across the grid.

The current design takes the speed.
That is a defensible trade for a sweep
whose results are read statistically.
It is a poor one for a result that has to be reproduced exactly.
The simulator cannot know which of the two a project does.

## What the trade means for a run

One thread is the setting for a run that has to be repeatable exactly,
such as the reduction of a model bug to a test case.

Many threads suit production sweeps,
where the output is one sample rather than the answer.
A run whose seed and thread count are recorded together
can at least be placed.
Such runs are compared by their summary series rather than tile by tile.

A thread count held fixed across a sweep
means that a difference between two runs of the sweep
can come only from the parameter that changed.
A varying thread count adds a second source of difference,
and that one is hard to separate from the first.

The steps for setting the thread count are in
[How to compile and run a model](../how-to-guides/compile-and-run-a-model.md#set-the-thread-count).

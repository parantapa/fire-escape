"""Input and output consistency checks for the Fire-ESCAPE simulator.

Used by ``tests/run-example1.py``.
The ``dims`` subcommand prints the grid and tick counts
implied by the input files as shell assignments.
The ``verify`` subcommand checks a simulator output file
against those inputs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import polars as pl

# Tick summary series written by every generated simulator,
# with the dtype kind the template declares for each.
TICK_SERIES: dict[str, str] = {
    "num_unburned": "i",
    "num_burning": "i",
    "num_burnt_out": "i",
    "num_ember_gen_sum": "i",
    "num_ember_recv_sum": "i",
    "num_flame_gen_sum": "f",
    "num_flame_recv_sum": "f",
}

# The fire_state encoding the template declares
# as Unburned, Burning and BurntOut.
UNBURNED = 0
BURNING = 1
BURNT_OUT = 2


# Callers turn an InputError into a counted failure or into an error line,
# so it never reaches the user as a traceback.
class InputError(Exception):
    """Raised when a file opens but its contents are not fit to keep checking.

    The message says which file is at fault and what is wrong with it.
    """


class ChecksAborted(Exception):
    """Raised to stop a run whose failure is already counted as a check."""


class Checker:
    """Collects check results and reports them as they are made."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0

    def check(self, ok: bool, message: str) -> bool:
        """Record and print one result, and return `ok` to guard a branch."""
        if ok:
            self.passed += 1
            print(f"[ ok ] {message}")
        else:
            self.failed += 1
            print(f"[fail] {message}")
        return ok

    def note(self, message: str) -> None:
        """Print an observation that counts as neither a pass nor a failure."""
        print(f"[info] {message}")

    def report(self) -> int:
        """Print a summary and return the number of failed checks."""
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed, {self.failed} failed")
        return self.failed


@dataclass(frozen=True)
class Inputs:
    """What the three input files say a run must look like."""

    rows: int
    cols: int
    num_ticks: int
    seed_state: np.ndarray
    num_seed_burning: int


@dataclass(frozen=True)
class StateScan:
    """Aggregates gathered in one pass over the saved state grids."""

    states_in_range: bool
    never_regresses: bool
    seed_tiles_alight: bool
    # Per tick counts of unburned, burning and burnt out tiles,
    # with shape (3, num_ticks).
    tallies: np.ndarray


def open_dataset(node: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """Return a named HDF5 dataset without reading its contents.

    Raises `InputError` when the member is not a dataset.
    """
    dataset = node[name]
    if not isinstance(dataset, h5py.Dataset):
        raise InputError(f"{node.file.filename}: {name} is not a dataset")
    return dataset


def read_dataset(node: h5py.File | h5py.Group, name: str) -> np.ndarray:
    """Read a named HDF5 dataset fully into memory."""
    return np.asarray(open_dataset(node, name)[...])


def read_group(fobj: h5py.File, name: str) -> h5py.Group:
    """Return a named HDF5 group.

    Raises `InputError` when the member is not a group.
    """
    group = fobj[name]
    if not isinstance(group, h5py.Group):
        raise InputError(f"{fobj.filename}: {name} is not a group")
    return group


def read_attr(node: h5py.Group, name: str) -> float:
    """Read a scalar HDF5 attribute, or NaN when it is absent."""
    if name not in node.attrs:
        return float("nan")
    return float(np.asarray(node.attrs[name]).item())


def all_true(values: np.typing.ArrayLike) -> bool:
    """Reduce an array of comparisons to a plain bool."""
    return bool(np.all(values))


def read_ticks(tick_file: Path) -> pl.DataFrame:
    """Read the per-tick global state."""
    return pl.read_parquet(tick_file)


def tick_keys(ticks_frame: pl.DataFrame, tick_file: Path) -> np.ndarray:
    """Return the sorted tick column of a tick frame.

    Raises `InputError` unless the column runs from 0 upwards
    with no gaps and no duplicates.
    """
    # The simulator drops any row whose tick falls outside the run,
    # which leaves those ticks with no wind at all.
    # A gapped or one-based column does that silently,
    # so it is caught here rather than after the run.
    if "tick" not in ticks_frame.columns:
        raise InputError(
            f"{tick_file}: no tick column (found {sorted(ticks_frame.columns)})"
        )
    ticks = np.sort(ticks_frame["tick"].to_numpy())
    if ticks.size == 0:
        raise InputError(f"{tick_file}: the tick column is empty")
    expected = np.arange(ticks.size)
    if not bool(np.array_equal(ticks, expected)):
        raise InputError(
            f"{tick_file}: the tick column is not 0 to {ticks.size - 1}"
            " without gaps or duplicates"
            f" (it runs from {int(ticks[0])} to {int(ticks[-1])}"
            f" over {ticks.size} rows)"
        )
    return ticks


def tick_count(tick_file: Path) -> int:
    """Return the number of ticks described by the tick data file."""
    return int(tick_keys(read_ticks(tick_file), tick_file).size)


def grid_shape(tile_file: Path) -> tuple[int, int]:
    """Return the (rows, cols) shape shared by the tile grids.

    Raises `InputError` when the file holds no two-dimensional dataset,
    or when the ones it holds disagree.
    """
    # A tile file can also carry members the model never names,
    # such as coordinate arrays or groups.
    # Only the two-dimensional datasets decide the shape.
    with h5py.File(tile_file, "r") as fobj:
        shapes: dict[str, tuple[int, ...]] = {}
        for name in fobj:
            node = fobj[str(name)]
            if isinstance(node, h5py.Dataset) and node.ndim == 2:
                shapes[str(name)] = node.shape
    if not shapes:
        raise InputError(f"{tile_file}: no two dimensional datasets found")
    distinct = set(shapes.values())
    if len(distinct) != 1:
        raise InputError(f"{tile_file}: inconsistent tile shapes {shapes}")
    rows, cols = distinct.pop()
    return int(rows), int(cols)


def read_dims(tile_file: Path, tick_file: Path) -> tuple[int, int, int]:
    """Return the (rows, cols, ticks) shape of a run over these inputs."""
    rows, cols = grid_shape(tile_file)
    return rows, cols, tick_count(tick_file)


def cmd_dims(args: argparse.Namespace) -> int:
    """Print grid and tick dimensions as shell variable assignments."""
    rows, cols, num_ticks = read_dims(args.tile_file, args.tick_file)
    print(f"NUM_ROWS={rows}")
    print(f"NUM_COLS={cols}")
    print(f"NUM_TICKS={num_ticks}")
    return 0


def verify_inputs(
    ck: Checker,
    tile_file: Path,
    seed_file: Path,
    tick_file: Path,
) -> Inputs:
    """Check that the three input files agree with each other.

    Raises `ChecksAborted` when the seed file holds no state member.
    Raises `InputError` when the tile file holds no consistent grid,
    when the tick file holds no usable tick column,
    or when the seed state is not a dataset.
    """
    rows, cols = grid_shape(tile_file)
    ticks_frame = read_ticks(tick_file)
    ticks = tick_keys(ticks_frame, tick_file)
    num_ticks = int(ticks.size)
    ck.note(f"grid is {rows} x {cols}, {num_ticks} ticks")

    with h5py.File(tile_file, "r") as fobj:
        names = set(fobj)
    ck.check(
        {"burn_time", "fuel", "moisture"} <= names,
        f"tile file holds burn_time, fuel and moisture (found {sorted(names)})",
    )

    with h5py.File(seed_file, "r") as fobj:
        if not ck.check("state" in fobj, "seed file holds the state dataset"):
            raise ChecksAborted
        seed_state = read_dataset(fobj, "state")
    ck.check(
        seed_state.shape == (rows, cols),
        f"seed grid {seed_state.shape} matches the tile grid ({rows}, {cols})",
    )
    num_seed_burning = int((seed_state == BURNING).sum())
    ck.check(num_seed_burning > 0, f"seed file has burning tiles ({num_seed_burning})")

    columns = set(ticks_frame.columns)
    ck.check(
        {"tick", "wind_speed", "wind_direction"} <= columns,
        f"tick file holds tick, wind_speed and wind_direction (found {sorted(columns)})",
    )
    # tick_keys has already raised on a gapped or duplicated column,
    # so this check can only record a pass.
    ck.check(
        bool(np.array_equal(ticks, np.arange(num_ticks))),
        f"tick keys cover 0 to {num_ticks - 1} without gaps or duplicates",
    )

    return Inputs(
        rows=rows,
        cols=cols,
        num_ticks=num_ticks,
        seed_state=seed_state,
        num_seed_burning=num_seed_burning,
    )


def verify_structure(
    ck: Checker,
    fobj: h5py.File,
    rows: int,
    cols: int,
    num_ticks: int,
) -> tuple[h5py.Dataset | None, dict[str, np.ndarray]]:
    """Check that the output file holds the state, the series and the runstats.

    The state dataset comes back open but unread,
    so a caller can walk it one tick at a time.
    It is None when the dataset is missing or misshapen.
    The mapping holds only the series of the expected length.
    """
    names = set(fobj)

    state: h5py.Dataset | None = None
    if ck.check("state" in names, "output holds the saved state dataset"):
        dataset = open_dataset(fobj, "state")
        if ck.check(
            dataset.shape == (num_ticks, rows, cols),
            f"state shape {dataset.shape} is ({num_ticks}, {rows}, {cols})",
        ):
            state = dataset

    missing = [name for name in TICK_SERIES if name not in names]
    ck.check(not missing, f"output holds all tick summary series (missing {missing})")

    series: dict[str, np.ndarray] = {}
    for name, kind in TICK_SERIES.items():
        if name not in names:
            continue
        data = read_dataset(fobj, name)
        right_length = data.shape == (num_ticks,)
        ck.check(
            right_length and data.dtype.kind == kind,
            f"{name} is a length {num_ticks} '{kind}' series"
            f" (got {data.shape} '{data.dtype}')",
        )
        # A short series is left out rather than returned,
        # because every check downstream
        # compares it against the others elementwise.
        if right_length:
            series[name] = data

    if ck.check("runstats" in names, "output holds the runstats group"):
        runstats = read_group(fobj, "runstats")
        alloc_bytes = read_attr(runstats, "total_alloc_bytes")
        ck.check(
            alloc_bytes > 0.0,
            f"runstats records a positive allocation size ({alloc_bytes / 1e9:.3f} GB)",
        )
        log_prob = read_attr(runstats, "log_prob")
        ck.check(
            bool(np.isfinite(log_prob)) and log_prob <= 0.0,
            f"runstats records a finite non positive log probability ({log_prob:.3f})",
        )
        duration = read_attr(runstats, "prog_duration")
        ck.check(
            bool(np.isfinite(duration)),
            f"runstats records the whole program duration ({duration:.3f} s)",
        )

    return state, series


def scan_state(state: h5py.Dataset, seed_state: np.ndarray) -> StateScan:
    """Gather the per-tick tallies and the state invariants in one pass.

    Peak memory is a few grids, not the whole dataset.
    """
    # Reading the whole dataset costs one byte per tile per tick,
    # which runs to gigabytes at the grid sizes the simulator handles.
    # Every whole array comparison over it costs as much again.
    # So each tick is read on its own,
    # and only the previous tick is held back for the ordering check.
    num_ticks = int(state.shape[0])
    num_tiles = int(state.shape[1] * state.shape[2])

    tallies = np.zeros((3, num_ticks), dtype=np.int64)
    states_in_range = True
    never_regresses = True
    seed_tiles_alight = True
    seed_is_burning = seed_state == BURNING

    previous: np.ndarray | None = None
    for tick in range(num_ticks):
        current = np.asarray(state[tick])
        tallies[0, tick] = int((current == UNBURNED).sum())
        tallies[1, tick] = int((current == BURNING).sum())
        tallies[2, tick] = int((current == BURNT_OUT).sum())
        # The three tallies only add up to the whole grid
        # when every tile holds one of the three known states.
        # The range check therefore needs no pass of its own.
        states_in_range = states_in_range and int(tallies[:, tick].sum()) == num_tiles
        if previous is None:
            seed_tiles_alight = all_true(current[seed_is_burning] != UNBURNED)
        else:
            # Comparing the two slices avoids the widening copy
            # that a difference of signed bytes would need.
            never_regresses = never_regresses and all_true(current >= previous)
        previous = current

    return StateScan(
        states_in_range=states_in_range,
        never_regresses=never_regresses,
        seed_tiles_alight=seed_tiles_alight,
        tallies=tallies,
    )


def verify_states(ck: Checker, scan: StateScan) -> None:
    """Check the saved state grids on their own terms."""
    ck.check(
        scan.states_in_range,
        "every saved state is Unburned, Burning or BurntOut",
    )
    ck.check(
        scan.never_regresses,
        "no tile ever moves backwards through Unburned, Burning, BurntOut",
    )
    ck.check(
        scan.seed_tiles_alight,
        "tiles burning in the seed file are alight at the first saved tick",
    )


def verify_series(
    ck: Checker,
    series: dict[str, np.ndarray],
    scan: StateScan,
    num_tiles: int,
    num_seed_burning: int,
) -> None:
    """Check the tick summary series against the saved state grids.

    Expects `series` to hold every name in `TICK_SERIES`.
    """
    reported = np.stack(
        [series["num_unburned"], series["num_burning"], series["num_burnt_out"]]
    )
    ck.check(
        all_true(scan.tallies == reported),
        "per tick state tallies match the saved state grids",
    )
    ck.check(
        all_true(reported.sum(axis=0) == num_tiles),
        f"the three state tallies sum to {num_tiles} tiles every tick",
    )
    ck.check(
        all_true(np.diff(series["num_unburned"]) <= 0),
        "the unburned count never increases",
    )
    ck.check(
        all_true(np.diff(series["num_burnt_out"]) >= 0),
        "the burnt out count never decreases",
    )
    ck.check(
        all_true(series["num_ember_recv_sum"] <= series["num_ember_gen_sum"]),
        "no tick receives more embers than it generated",
    )
    ck.check(
        all_true(np.isfinite(series["num_flame_gen_sum"]))
        and all_true(series["num_flame_gen_sum"] >= 0.0)
        and all_true(np.isfinite(series["num_flame_recv_sum"]))
        and all_true(series["num_flame_recv_sum"] >= 0.0),
        "flame generated and received are finite and non negative",
    )

    burned = int(series["num_burning"][-1] + series["num_burnt_out"][-1])
    # Spread beyond the seed tiles is the usual outcome, not an invariant.
    # Ignition is drawn at random,
    # and a seed sitting in wet ground or in thin fuel can burn out alone.
    # A correct simulator is free to produce such a run,
    # so the extent of the fire is reported rather than checked.
    ck.note(
        f"the fire reached {burned} tiles"
        " burning or burnt out at the last tick,"
        f" from {num_seed_burning} seed tiles"
    )
    ck.note(
        f"final tally: {int(series['num_unburned'][-1])} unburned,"
        f" {int(series['num_burning'][-1])} burning,"
        f" {int(series['num_burnt_out'][-1])} burnt out"
        f" ({100.0 * burned / num_tiles:.2f}% of the grid)"
    )
    ck.note(
        f"embers generated: {int(series['num_ember_gen_sum'].sum())},"
        f" received: {int(series['num_ember_recv_sum'].sum())};"
        f" flame generated: {float(series['num_flame_gen_sum'].sum()):.1f},"
        f" received: {float(series['num_flame_recv_sum'].sum()):.1f}"
    )


def run_checks(
    ck: Checker,
    tile_file: Path,
    seed_file: Path,
    tick_file: Path,
    output_file: Path,
) -> None:
    """Run every check over one set of input files and one output file."""
    inputs = verify_inputs(ck, tile_file, seed_file, tick_file)

    with h5py.File(output_file, "r") as fobj:
        state, series = verify_structure(
            ck, fobj, inputs.rows, inputs.cols, inputs.num_ticks
        )
        # A seed grid of the wrong shape cannot be laid over the saved state.
        # verify_inputs already counted that as a failure,
        # so the grid checks are skipped rather than allowed to crash.
        if state is None or inputs.seed_state.shape != (inputs.rows, inputs.cols):
            return
        scan = scan_state(state, inputs.seed_state)

    verify_states(ck, scan)
    if len(series) == len(TICK_SERIES):
        verify_series(
            ck,
            series,
            scan,
            inputs.rows * inputs.cols,
            inputs.num_seed_burning,
        )


def check_all(
    tile_file: Path,
    seed_file: Path,
    tick_file: Path,
    output_file: Path,
) -> int:
    """Check an output file against its inputs.

    Prints each result and a summary to stdout,
    and returns the number of failed checks.
    An `InputError` counts as a failed check rather than propagating.
    """
    ck = Checker()
    try:
        run_checks(ck, tile_file, seed_file, tick_file, output_file)
    except ChecksAborted:
        # The failure that stopped the run is already counted.
        pass
    except InputError as error:
        ck.check(False, str(error))
    return ck.report()


def cmd_verify(args: argparse.Namespace) -> int:
    """Check a simulator output file against the inputs that produced it."""
    failed = check_all(args.tile_file, args.seed_file, args.tick_file, args.output_file)
    return 1 if failed else 0


def main() -> int:
    """Run the requested subcommand and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dims = sub.add_parser("dims", help="print input dimensions as shell assignments")
    dims.add_argument("--tile-file", type=Path, required=True)
    dims.add_argument("--tick-file", type=Path, required=True)
    dims.set_defaults(func=cmd_dims)

    verify = sub.add_parser("verify", help="check an output file against the inputs")
    verify.add_argument("--tile-file", type=Path, required=True)
    verify.add_argument("--seed-file", type=Path, required=True)
    verify.add_argument("--tick-file", type=Path, required=True)
    verify.add_argument("--output-file", type=Path, required=True)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    command: Callable[[argparse.Namespace], int] = args.func
    try:
        return command(args)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

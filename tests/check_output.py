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

UNBURNED = 0
BURNING = 1
BURNT_OUT = 2


class Checker:
    """Collects check results and reports them as they are made."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0

    def check(self, ok: bool, message: str) -> bool:
        if ok:
            self.passed += 1
            print(f"[ ok ] {message}")
        else:
            self.failed += 1
            print(f"[fail] {message}")
        return ok

    def note(self, message: str) -> None:
        print(f"[info] {message}")

    def report(self) -> int:
        """Print a summary and return the number of failed checks."""
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed, {self.failed} failed")
        return self.failed


def read_dataset(node: h5py.File | h5py.Group, name: str) -> np.ndarray:
    """Read a named HDF5 dataset fully into memory."""
    dataset = node[name]
    if not isinstance(dataset, h5py.Dataset):
        raise SystemExit(f"{node.file.filename}: {name} is not a dataset")
    return np.asarray(dataset[...])


def read_group(fobj: h5py.File, name: str) -> h5py.Group:
    """Read a named HDF5 group."""
    group = fobj[name]
    if not isinstance(group, h5py.Group):
        raise SystemExit(f"{fobj.filename}: {name} is not a group")
    return group


def read_attr(node: h5py.Group, name: str) -> float:
    """Read a scalar HDF5 attribute, or NaN when it is absent."""
    if name not in node.attrs:
        return float("nan")
    return float(np.asarray(node.attrs[name]).item())


def all_true(values: Any) -> bool:
    """Reduce an array of comparisons to a plain bool."""
    return bool(np.all(values))


def read_ticks(tick_file: Path) -> pl.DataFrame:
    """Read the per tick global state."""
    return pl.read_parquet(tick_file)


def tick_count(tick_file: Path) -> int:
    """Return the number of ticks described by the tick data file."""
    ticks = read_ticks(tick_file)["tick"].to_numpy()
    return int(ticks.max()) + 1


def grid_shape(tile_file: Path) -> tuple[int, int]:
    """Return the (rows, cols) shape shared by the tile datasets."""
    with h5py.File(tile_file, "r") as fobj:
        shapes = {str(name): read_dataset(fobj, str(name)).shape for name in fobj}
    if not shapes:
        raise SystemExit(f"{tile_file}: no datasets found")
    distinct = set(shapes.values())
    if len(distinct) != 1:
        raise SystemExit(f"{tile_file}: inconsistent tile shapes {shapes}")
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
) -> tuple[int, int, int]:
    """Check that the three input files agree with each other."""
    rows, cols, num_ticks = read_dims(tile_file, tick_file)
    ck.note(f"grid is {rows} x {cols}, {num_ticks} ticks")

    with h5py.File(tile_file, "r") as fobj:
        names = set(fobj)
    ck.check(
        {"burn_time", "fuel", "moisture"} <= names,
        f"tile file holds burn_time, fuel and moisture (found {sorted(names)})",
    )

    with h5py.File(seed_file, "r") as fobj:
        if not ck.check("state" in fobj, "seed file holds the state dataset"):
            raise SystemExit(1)
        seed_state = read_dataset(fobj, "state")
    ck.check(
        seed_state.shape == (rows, cols),
        f"seed grid {seed_state.shape} matches the tile grid ({rows}, {cols})",
    )
    num_seed_burning = int((seed_state == BURNING).sum())
    ck.check(num_seed_burning > 0, f"seed file has burning tiles ({num_seed_burning})")

    ticks_frame = read_ticks(tick_file)
    columns = set(ticks_frame.columns)
    ck.check(
        {"tick", "wind_speed", "wind_direction"} <= columns,
        f"tick file holds tick, wind_speed and wind_direction (found {sorted(columns)})",
    )
    ticks = ticks_frame["tick"].sort().to_numpy()
    ck.check(
        bool(np.array_equal(ticks, np.arange(num_ticks))),
        f"tick keys cover 0 to {num_ticks - 1} without gaps or duplicates",
    )

    return rows, cols, num_ticks


def verify_structure(
    ck: Checker,
    fobj: h5py.File,
    rows: int,
    cols: int,
    num_ticks: int,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Check that the output file holds the datasets the model asked for."""
    names = set(fobj)

    state: np.ndarray | None = None
    if ck.check("state" in names, "output holds the saved state dataset"):
        state = read_dataset(fobj, "state")
        ck.check(
            state.shape == (num_ticks, rows, cols),
            f"state shape {state.shape} is ({num_ticks}, {rows}, {cols})",
        )

    missing = [name for name in TICK_SERIES if name not in names]
    ck.check(not missing, f"output holds all tick summary series (missing {missing})")

    series: dict[str, np.ndarray] = {}
    for name, kind in TICK_SERIES.items():
        if name not in names:
            continue
        data = read_dataset(fobj, name)
        series[name] = data
        ck.check(
            data.shape == (num_ticks,) and data.dtype.kind == kind,
            f"{name} is a length {num_ticks} '{kind}' series"
            f" (got {data.shape} '{data.dtype}')",
        )

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


def verify_states(ck: Checker, state: np.ndarray, seed_state: np.ndarray) -> None:
    """Check the saved state grids on their own terms."""
    ck.check(
        all_true(np.isin(state, (UNBURNED, BURNING, BURNT_OUT))),
        "every saved state is Unburned, Burning or BurntOut",
    )
    ck.check(
        all_true(np.diff(state.astype(np.int16), axis=0) >= 0),
        "no tile ever moves backwards through Unburned, Burning, BurntOut",
    )
    ck.check(
        all_true(state[0][seed_state == BURNING] != UNBURNED),
        "tiles burning in the seed file are alight at the first saved tick",
    )


def verify_series(
    ck: Checker,
    series: dict[str, np.ndarray],
    state: np.ndarray,
    seed_state: np.ndarray,
) -> None:
    """Check the tick summary series against the saved state grids."""
    num_tiles = int(state.shape[1] * state.shape[2])

    counted = np.stack(
        [
            (state == UNBURNED).sum(axis=(1, 2)),
            (state == BURNING).sum(axis=(1, 2)),
            (state == BURNT_OUT).sum(axis=(1, 2)),
        ]
    )
    reported = np.stack(
        [series["num_unburned"], series["num_burning"], series["num_burnt_out"]]
    )
    ck.check(
        all_true(counted == reported),
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

    num_seed_burning = int((seed_state == BURNING).sum())
    burned = int(series["num_burning"][-1] + series["num_burnt_out"][-1])
    ck.check(
        burned > num_seed_burning,
        f"the fire spread beyond the {num_seed_burning} seed tiles"
        f" ({burned} tiles burning or burnt out at the last tick)",
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


def check_all(
    tile_file: Path,
    seed_file: Path,
    tick_file: Path,
    output_file: Path,
) -> int:
    """Check an output file against its inputs.

    Returns the number of failed checks.
    """
    ck = Checker()
    rows, cols, num_ticks = verify_inputs(ck, tile_file, seed_file, tick_file)

    with h5py.File(seed_file, "r") as fobj:
        seed_state = read_dataset(fobj, "state")

    with h5py.File(output_file, "r") as fobj:
        state, series = verify_structure(ck, fobj, rows, cols, num_ticks)

    if state is not None and state.shape == (num_ticks, rows, cols):
        verify_states(ck, state, seed_state)
        if len(series) == len(TICK_SERIES):
            verify_series(ck, series, state, seed_state)

    return ck.report()


def cmd_verify(args: argparse.Namespace) -> int:
    """Check a simulator output file against the inputs that produced it."""
    failed = check_all(args.tile_file, args.seed_file, args.tick_file, args.output_file)
    return 1 if failed else 0


def main() -> int:
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
    return command(args)


if __name__ == "__main__":
    sys.exit(main())

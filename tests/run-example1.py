#!/usr/bin/env python3
"""End to end test of the OpenMP CPU backend.

Compiles an FFSL model with ffsc,
builds the generated C++ project with Conan,
runs the simulator over the t1 test dataset,
and checks the output file against the inputs.

Run with --help for the argument list.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn, Sequence

import click

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from check_output import check_all, read_dims
except ImportError as error:  # pragma: no cover - depends on the environment
    raise SystemExit(
        f"error: the checks need h5py and polars ({error});"
        f" run 'pip install -e {REPO_ROOT}[dev]'"
    )


def log(message: str) -> None:
    """Announce the step that is about to run."""
    click.secho(f"\n=== {message}", bold=True)


def die(message: str) -> NoReturn:
    """Abort the test with an error message."""
    raise SystemExit(f"error: {message}")


def run(
    command: Sequence[str | Path],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command, letting its output through, and abort if it fails."""
    argv = [str(part) for part in command]
    click.echo("$ " + shlex.join(argv))
    completed = subprocess.run(argv, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        die(f"{argv[0]} exited with status {completed.returncode}")


def check_prerequisites(input_files: Sequence[Path]) -> None:
    """Check that the tools and the input files needed by the test are there."""
    if shutil.which("ffsc") is None:
        die(f"ffsc not found; run 'pip install .' in {REPO_ROOT}")
    if shutil.which("conan") is None:
        die("conan not found; the C++ project cannot be built")
    for input_file in input_files:
        if not input_file.is_file():
            die(f"missing input file {input_file}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--model",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="FFSL model file to compile.",
)
@click.option(
    "-d",
    "--data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory holding the t1-* data files.",
)
@click.option(
    "-w",
    "--work-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for the generated project and the run output.",
)
@click.option(
    "--build-type",
    default="Release",
    show_default=True,
    envvar="BUILD_TYPE",
    help="Conan and CMake build type.",
)
@click.option(
    "--cppstd",
    default="gnu23",
    show_default=True,
    envvar="CPPSTD",
    help="compiler.cppstd setting for Conan.",
)
@click.option(
    "--threads",
    default=4,
    show_default=True,
    type=int,
    envvar="OMP_NUM_THREADS",
    help="Number of OpenMP threads used by the simulator.",
)
@click.option(
    "--skip-build",
    is_flag=True,
    envvar="SKIP_BUILD",
    help="Reuse an already built simulator binary.",
)
@click.argument("simulator_args", nargs=-1, type=click.UNPROCESSED)
def main(
    model: Path,
    data_dir: Path,
    work_dir: Path,
    build_type: str,
    cppstd: str,
    threads: int,
    skip_build: bool,
    simulator_args: tuple[str, ...],
) -> None:
    """Compile, build, run and check a model against the t1 test dataset.

    Arguments given after -- are passed on to the simulator,
    so run time config options can be exercised like this:

    \b
        tests/run-example1.py -m examples/example1.ffsl \\
            -d /mnt/data/fire-escape/0019_01316 \\
            -w build/test-example1 \\
            -- --config-base-ember-rate 0.9
    """
    tile_file = data_dir / "t1-tiles.h5"
    seed_file = data_dir / "t1-seeds.h5"
    tick_file = data_dir / "t1-tick.parquet"
    output_file = work_dir / "t1-output.h5"
    simulator = work_dir / "build" / build_type / "simulator"

    log("Checking prerequisites")
    check_prerequisites([model, tile_file, seed_file, tick_file])
    click.echo(f"model:  {model}")
    click.echo(f"data:   {data_dir}")
    click.echo(f"work:   {work_dir}")

    log("Compiling the model to a C++ project")
    run(["ffsc", "compile", "-i", model, "-o", work_dir])

    if skip_build:
        log("Skipping the build as requested")
        if not os.access(simulator, os.X_OK):
            die(f"no simulator at {simulator}; drop --skip-build")
    else:
        log("Building the simulator")
        # The generated CMakeLists asks for C++23,
        # so Conan has to agree with it
        # rather than impose the cppstd of the default profile.
        run(
            [
                "conan",
                "build",
                ".",
                "--build=missing",
                "-s",
                f"build_type={build_type}",
                "-s",
                f"compiler.cppstd={cppstd}",
            ],
            cwd=work_dir,
        )
        if not os.access(simulator, os.X_OK):
            die(f"the build produced no simulator at {simulator}")

    log("Reading the input dimensions")
    num_rows, num_cols, num_ticks = read_dims(tile_file, tick_file)
    click.echo(f"NUM_ROWS={num_rows} NUM_COLS={num_cols} NUM_TICKS={num_ticks}")

    log(f"Running the simulator on {threads} threads")
    output_file.unlink(missing_ok=True)
    run(
        [
            simulator,
            "--num-ticks",
            str(num_ticks),
            "--num-rows",
            str(num_rows),
            "--num-cols",
            str(num_cols),
            "--tile-file",
            tile_file,
            "--seed-file",
            seed_file,
            "--tick-file",
            tick_file,
            "--output-file",
            output_file,
            *simulator_args,
        ],
        env=os.environ | {"OMP_NUM_THREADS": str(threads)},
    )

    log("Checking the output")
    failed = check_all(tile_file, seed_file, tick_file, output_file)
    if failed:
        die(f"{failed} checks failed")

    log(f"PASS: output written to {output_file}")


if __name__ == "__main__":
    sys.exit(main())

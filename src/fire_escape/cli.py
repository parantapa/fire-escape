"""Command line interface."""

from pathlib import Path

import click
import rich

from .templates import render
from .parser import parse


@click.group()
def cli():
    """Forest fire simulator language (FFSL) compiler."""


@cli.command()
@click.option(
    "-i",
    "--input-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="FFSL simulator code.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="C++ project directory.",
)
def compile(input_file: Path, output_dir: Path):
    """Compile the FFSL code to a C++ project."""
    output_dir.mkdir(exist_ok=True, parents=True, mode=0o755)

    module = input_file.name
    print(f"Compiling module: {module}")

    # print(.pretty())
    rich.print(parse(input_file.read_text()))

    with open(output_dir / "main.cpp", "wt") as fobj:
        code = render("openmp-cpu:main.cpp", module=module)
        fobj.write(code)

    with open(output_dir / "CMakeLists.txt", "wt") as fobj:
        code = render("openmp-cpu:CMakeLists.txt", module=module)
        fobj.write(code)


def main():
    cli()

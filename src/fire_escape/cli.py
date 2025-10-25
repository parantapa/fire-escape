"""Command line interface."""

from pathlib import Path

import click
import rich

from .parser import parse
from .templates import render
from .codegen import codegen_openmp_cpu


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

    source = parse(str(input_file), input_file.read_text())
    rich.print(source)

    with open(output_dir / "main.cpp", "wt") as fobj:
        code = codegen_openmp_cpu(source)
        fobj.write(code)

    with open(output_dir / "CMakeLists.txt", "wt") as fobj:
        code = render("openmp-cpu:CMakeLists.txt", module=module)
        fobj.write(code)


def main():
    cli()

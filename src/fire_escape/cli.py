"""Command line interface."""

import sys

import click

from .codegen_openmp_cpu import compile_cmd
from .error import CodeError


@click.group()
def cli() -> None:
    """Forest fire simulator language (FFSL) compiler."""


cli.add_command(compile_cmd)


def main() -> None:
    """Run the command line interface.

    Exits with status 1 on an error in the FFSL source being compiled.
    """
    try:
        cli()
    except CodeError as error:
        # A `CodeError` prints as one line, without a traceback.
        # See the developer notes on compiler faults and model faults.
        print(error, file=sys.stderr)
        raise SystemExit(1)

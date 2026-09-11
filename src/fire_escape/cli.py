"""Command line interface."""

import sys

import click

from .codegen_openmp_cpu import compile_cmd
from .error import CodeError


@click.group()
def cli():
    """Forest fire simulator language (FFSL) compiler."""


cli.add_command(compile_cmd)


def main():
    """Run the command line interface.

    A CodeError reports a mistake in the FFSL source being compiled,
    so it is printed as a plain located message.
    Any other exception reports a mistake in the compiler itself,
    and it keeps its traceback.
    """
    try:
        cli()
    except CodeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)

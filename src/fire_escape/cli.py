"""Command line interface."""

import click

from .codegen_openmp_cpu import compile_cmd


@click.group()
def cli():
    """Forest fire simulator language (FFSL) compiler."""


cli.add_command(compile_cmd)


def main():
    cli()

"""Command line interface."""

import click

from .codegen_openmp_cpu import openmp_cpu


@click.group()
def cli():
    """Forest fire simulator language (FFSL) compiler."""


cli.add_command(openmp_cpu)


def main():
    cli()

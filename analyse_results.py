import os

import click
import keras
import hickle as hkl 

@click.command(help="Analyse model results.")
@click.option("-w", "--weights-path", prompt=True, type=str)
@click.option("-t", "--training-path", prompt=True, type=str)
@click.option("-v", "--validation-path", prompt=True, type=str)
def main(
    weights_path: str,
    training_path: str,
    validation_path: str,
) -> None:

    

import json

import click
import numpy as np

import src.data
from src.metrics import compute_metrics


@click.command()
@click.argument("file", type=click.File("rb"))
def main(file):
    predictions = np.load(file).squeeze()
    N, _ = predictions.shape

    lookup = {e.n_series: e for e in src.data.info.values()}
    INFO = lookup[N]

    dm = src.data.M4DataModule(INFO.frequency)

    print(INFO.frequency)
    print(json.dumps(compute_metrics(dm, predictions), indent=4))


if __name__ == "__main__":
    main()

# Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting
Public implementation of "Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting".

## Paper
https://arxiv.org/abs/2208.14236 

## Run code

1. `make pull`
2. Run `./docker_run python scripts/run.py`

### Command line options
The default setting can be found in [conf.py](/src/conf.py).
Options can be set via the command line, for instance:
`./docker_run python scripts/run.py freq=YEARLY model=TRANSFORMER persistence_init=NONE d_model=64 layer=POST_NORM pos_enc=SINUSOIDAL` will run a "regular" Transformer model, without any of the proposed changes from our paper.

It is also possible to run a range of experiments by using Hydra's multirun feature:
`./docker_run python scripts/run.py -m freq=YEARLY,QUARTERLY,MONTHLY,WEEKLY,DAILY,HOURLY` to run a small model for each data frequency.

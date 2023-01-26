#!/usr/bin/env python3
import warnings

import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.autoformer.lightning import AutoformerLightningModule
from src.conf import Config, ModelEnum, TrainingSetupEnum
from src.data import M4DataModule
from src.informer.lightning import InformerLightningModule
from src.transformer.lightning import TransformerLightningModule

warnings.filterwarnings("ignore", category=pl.utilities.warnings.LightningDeprecationWarning)


def create_trainer(max_epochs: int = 500, patience: int = 8) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=128,
        log_every_n_steps=128,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="checkpoints",
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=patience,
                verbose=True,
                mode="min",
                strict=False,
                check_finite=True,
            ),
        ],
        logger=[
            CSVLogger(
                "logs",
                name="",
                version="",
            ),
            TensorBoardLogger(
                "tensorboard",
                name="",
                version="",
                default_hp_metric=False,
            ),
        ],
        num_sanity_val_steps=0,
        gpus="0,",
    )


@hydra.main(config_path=None, config_name="config")
def main(config: Config):
    seed_everything(config.seed)

    data_module = M4DataModule(config.freq.value)
    data_module.prepare_data()
    data_module.setup()

    model: pl.LightningModule
    if config.model is ModelEnum.TRANSFORMER:
        model = TransformerLightningModule(
            data_module,
            d_model=config.d_model,
            persistence_init=config.persistence_init,
            pos_enc=config.pos_enc,
            layer=config.layer,
        )
    elif config.model is ModelEnum.INFORMER:
        model = InformerLightningModule(data_module)
    elif config.model is ModelEnum.AUTOFORMER:
        model = AutoformerLightningModule(data_module)
    else:
        raise ValueError

    if config.training_setup is TrainingSetupEnum.EARLY_STOPPING:
        trainer = create_trainer()
    elif config.training_setup is TrainingSetupEnum.MAX_EPOCH:
        trainer = create_trainer(max_epochs=100, patience=1000)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module, ckpt_path="best", verbose=True)


if __name__ == "__main__":
    main()

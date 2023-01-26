from pathlib import Path

import apex
import numpy as np

import pytorch_lightning as pl
import torch

from src.conf import LayerEnum, PersistenceInitEnum, PositionalEncodingEnum
from src.data import Batch, M4DataModule
from src.metrics import compute_metrics

from .transformer import DecoderTransformer


class TransformerLightningModule(pl.LightningModule):
    def forward(self, x, mean, mask, n):
        # teacher forcing
        x = x / mean
        x.log_()
        x = self.transformer(x, mask=mask)
        x = x[:-1]
        x = x[-n:]
        x += mean.log()
        x.exp_()
        return x

    def ar_forward(self, x, mean, mask, n):
        # autoregressive predictions,
        # assumes last n elements of x are to be filled in
        x = x / mean
        x.log_()
        for size in range(x.size(0) - n, x.size(0)):
            one_step = self.transformer(x[:size], mask=mask[:, :size, :size])[-1:]
            x[size : size + 1] = one_step

        x = x[-n:]
        x += mean.log()
        x.exp_()
        return x

    def training_step(self, batch: Batch, batch_idx: int):  # type: ignore[override]
        loss = self._loss_step(batch).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def _loss_step(self, batch: Batch):
        outputs = self(batch.data, batch.mean, batch.mask, batch.targets.size(0))

        s_naive_MAE = torch.index_select(self.s_naive_MAE, 1, batch.idx)
        loss = torch.abs(batch.targets - outputs)
        loss = loss / s_naive_MAE

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):  # type: ignore[override]
        return {
            "loss": self._loss_step(batch),
        }

    def test_step(self, batch: Batch, batch_idx: int):  # type: ignore[override]
        return {
            "preds": self.ar_forward(batch.data, batch.mean, batch.mask, batch.targets.size(0)),
        }

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o["loss"] for o in outputs], dim=1)
        loss = loss.mean()
        self.log("val_loss", loss)

    def test_epoch_end(self, outputs):
        preds = torch.cat([o["preds"] for o in outputs], dim=1)
        preds = preds.transpose(0, 1).squeeze().cpu().numpy()

        self.log_dict(compute_metrics(self.dm, preds))

        path = Path("activations") / f"test_predictions_epoch{self.current_epoch}.npy"
        path.parent.mkdir(exist_ok=True)
        np.save(path, preds)

    def __init__(
        self,
        data_module: M4DataModule,
        *,
        d_model: int,
        persistence_init: PersistenceInitEnum,
        pos_enc: PositionalEncodingEnum,
        layer: LayerEnum,
    ):
        super().__init__()

        self.dm: M4DataModule = data_module

        self.s_naive_MAE: torch.Tensor
        self.register_buffer(
            "s_naive_MAE",
            torch.from_numpy(data_module.statistics["s_naive_MAE"].values[None, :, None]),
            persistent=False,
        )

        self.transformer: DecoderTransformer = DecoderTransformer(
            d_model=d_model,
            persistence_init=persistence_init,
            pos_enc=pos_enc,
            layer=layer,
        )

    def configure_optimizers(self):
        return apex.optimizers.fused_lamb.FusedLAMB(
            self.parameters(),
            lr=1.0e-3,
            betas=(0.9, 0.999),
            eps=1.0e-6,
            weight_decay=0.0,
            max_grad_norm=10.0,
            bias_correction=True,
        )

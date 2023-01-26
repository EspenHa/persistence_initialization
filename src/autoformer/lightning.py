from pathlib import Path

import apex
import numpy as np

import pytorch_lightning as pl
import torch

from src.data import Batch, M4DataModule
from src.metrics import compute_metrics

from .autoformer import Autoformer


class AutoformerLightningModule(pl.LightningModule):
    def forward(self, x_enc, x_dec, mean, n):
        x_enc = x_enc / mean
        x_enc.log_()

        x_dec = x_dec / mean
        x_dec.log_()
        x_dec[-n:] = 0  # make sure targets are 0 so we don't cheat

        x = self.autoformer(x_enc, x_dec)

        x = x[-n:]
        x += mean.log()
        x.exp_()
        return x

    def _loss_step(self, batch: Batch):
        target_size = batch.targets.size(0)

        enc_inputs = batch.data[:-target_size]
        dec_inputs = batch.data[-2 * target_size :]

        outputs = self(enc_inputs, dec_inputs, batch.mean, target_size)

        s_naive_MAE = torch.index_select(self.s_naive_MAE, 1, batch.idx)
        loss = torch.abs(batch.targets - outputs)
        loss = loss / s_naive_MAE

        return loss, outputs

    def training_step(self, batch: Batch, batch_idx: int):  # type: ignore[override]
        loss, _ = self._loss_step(batch)
        loss = loss.mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):  # type: ignore[override]
        loss, _ = self._loss_step(batch)
        return {"loss": loss}

    def test_step(self, batch: Batch, batch_idx: int):  # type: ignore[override]
        _, preds = self._loss_step(batch)
        return {"preds": preds}

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

    def __init__(self, data_module: M4DataModule):
        super().__init__()

        self.dm: M4DataModule = data_module

        self.s_naive_MAE: torch.Tensor
        self.register_buffer(
            "s_naive_MAE",
            torch.from_numpy(data_module.statistics["s_naive_MAE"].values[None, :, None]),
            persistent=False,
        )

        self.autoformer = Autoformer(
            label_len=2 * data_module.INFO.outsample_size,
            pred_len=data_module.INFO.outsample_size,
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

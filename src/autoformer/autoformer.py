import torch
import torch.nn as nn

from .autocorrelation import AutoCorrelation, AutoCorrelationLayer
from .embed import DataEmbedding_wo_pos
from .enc_dec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)


class Autoformer(nn.Module):
    def __init__(
        self,
        label_len: int,
        pred_len: int,
        *,
        d_model: int = 32,
        e_layers: int = 2,
        d_layers: int = 2,
        n_heads: int = 4,
        factor: int = 5,
    ):
        super().__init__()

        self.label_len = label_len
        self.pred_len = pred_len

        self.decomp = series_decomp(kernel_size=factor)

        self.enc_embedding = DataEmbedding_wo_pos(1, d_model, dropout=0.0)
        self.dec_embedding = DataEmbedding_wo_pos(1, d_model, dropout=0.0)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor=factor,
                            attention_dropout=0.0,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff=4 * d_model,
                    moving_avg=5,
                    dropout=0.0,
                    activation="gelu",
                )
                for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=0.0,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=0.0,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    1,
                    d_ff=4 * d_model,
                    moving_avg=factor,
                    dropout=0.0,
                    activation="gelu",
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True),
        )

    def forward(
        self,
        x_enc,
        x_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        x_enc.nan_to_num_()
        x_dec.nan_to_num_()

        enc_out = self.enc_forward(x_enc, enc_self_mask)
        dec_out = self.dec_forward(enc_out, x_enc, x_dec, dec_self_mask, dec_enc_mask)

        return dec_out

    def enc_forward(
        self,
        x_enc,
        enc_self_mask=None,
    ):
        x_enc = x_enc.transpose(0, 1)

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        return enc_out

    def dec_forward(
        self,
        enc_out,
        x_enc,
        x_dec,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        x_enc = x_enc.transpose(0, 1)
        x_dec = x_dec.transpose(0, 1)

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len :, :], zeros], dim=1)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        dec_out = dec_out.transpose(0, 1)
        return dec_out

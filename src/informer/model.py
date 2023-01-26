import torch

from .attn import AttentionLayer, ProbAttention
from .decoder import Decoder, DecoderLayer
from .embed import DataEmbedding
from .encoder import Encoder, EncoderLayer


class Informer(torch.nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 32,
        e_layers: int = 2,
        d_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()

        self.enc_embedding = DataEmbedding(1, d_model, dropout=0.0, conv=True)
        self.dec_embedding = DataEmbedding(1, d_model, dropout=0.0, conv=True)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            factor=5,
                            attention_dropout=0.0,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads=n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff=4 * d_model,
                    dropout=0.0,
                    activation="gelu",
                )
                for l in range(e_layers)
            ],
            None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            factor=5,
                            attention_dropout=0.0,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads=n_heads,
                        mix=True,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor=5,
                            attention_dropout=0.0,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads=n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff=4 * d_model,
                    dropout=0.0,
                    activation="gelu",
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.projection = torch.nn.Linear(d_model, 1, bias=True)

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
        dec_out = self.dec_forward(enc_out, x_dec, dec_self_mask, dec_enc_mask)

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
        x_dec,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        x_dec = x_dec.transpose(0, 1)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        dec_out = dec_out.transpose(0, 1)

        return dec_out

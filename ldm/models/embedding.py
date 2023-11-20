import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.embedding.modules import Encoder, Decoder

class XrayEmbedding(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 image_key='xray',
                 label_key='image',
                 remap=None,
                 ignore_keys=[],
                 monitor=None,
                 scheduler_config=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scheduler_config = scheduler_config

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, x_key, y_key):
        x, y = batch[x_key], batch[y_key]
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.get_input(batch, self.image_key, self.label_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        q_loss = qloss.mean()

        # L2 loss
        rec_loss = F.mse_loss(y.contiguous(), xrec.contiguous()).mean()

        total_loss = rec_loss + q_loss
        self.log(f"train/rec_loss", total_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input(batch, self.image_key, self.label_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        q_loss = qloss.mean()

        # L2 loss
        rec_loss = F.mse_loss(y.contiguous(), xrec.contiguous()).mean()

        total_loss = rec_loss + q_loss
        self.log(f"val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.encoder.parameters()) + \
                 list(self.decoder.parameters()) + \
                 list(self.quantize.parameters()) + \
                 list(self.quant_conv.parameters()) + \
                 list(self.post_quant_conv.parameters())

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt], scheduler

        return [opt]

class XrayEmbeddingInterface(XrayEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec
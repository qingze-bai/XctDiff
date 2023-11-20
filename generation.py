import argparse
import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.lidc_idri import get_loader
from ldm.util import instantiate_from_config
from ldm.modules.loggers.logger import ImageLogger

import torch
import nibabel as nib
import os
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="the config path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="the checkpoint file path",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="the input X-ray image",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="the generation step (default 50)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="generation eta (default 0.0)",
    )
    return parser.parse_args()


if __name__ == '__main__':

    hparams = get_parser()

    cfg = OmegaConf.load(hparams.cfg_path)

    xray = np.array(Image.open(hparams.input_path).convert("L").resize((128, 128)))
    xray = (xray.transpose(1, 0)[:,::-1]) / 255
    xray = torch.tensor(xray[np.newaxis, np.newaxis, ...], dtype=torch.float32).cuda()

    loader = get_loader(cfg.data)

    # loading model from cfg and load checkpoint
    cfg.model.params.ckpt_path = hparams.ckpt_path
    model = instantiate_from_config(cfg.model)
    model = model.cuda()

    # DDIM sampler
    ddim_sampler = DDIMSampler(model)
    ddim_steps = hparams.ddim_steps
    eta = hparams.eta


    with torch.no_grad():
        # print(xray.shape)

        cond = model.get_learned_conditioning(xray)
        B, C, H, W, D = cond.shape
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size=B, shape=(C, H, W, D), conditioning=cond,
                                                     verbose=False, eta=eta)
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clip(x_samples, -1, 1)

        nib.save(nib.Nifti1Image(x_samples[0, 0].detach().cpu().numpy().astype(np.float32), np.eye(4)),
                                  os.path.join('output', "result.nii.gz"))


        # print(x_samples.shape)
        # for i, batch in enumerate(loader[1]):
        #     cond = model.get_learned_conditioning(xray)
        #     B, C, H, W, D = c.shape
        #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size=B, shape=(C, H, W, D), conditioning=cond, verbose=False, eta=eta)
        #     # x_samples = model.decode_first_stage(samples)
        #     # x_samples = torch.clip(x_samples, -1, 1)
        #
        #     for i in range(B):
        #         cur_affine = affine[i].detach().cpu().numpy()
        #         nib.save(nib.Nifti1Image(x_samples[i,0].detach().cpu().numpy().astype(np.float32), cur_affine),
        #                  os.path.join('output', filename[i] + ".nii.gz"))
        #
        #
        #
        #     print(x_samples.shape, x_samples.min(), x_samples.max())

import argparse
import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.lidc_idri import get_loader
from ldm.util import instantiate_from_config
from ldm.modules.loggers.logger import ImageLogger

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="the config path",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200000,
        help="number of training iterations"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="training gpu number"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="accumulate grad batches"
    )
    parser.add_argument(
        "--every_n_train_steps",
        type=int,
        default=10000,
        help="frequency for saving checkpoint"
    )


    return parser.parse_args()


if __name__ == '__main__':

    hparams = get_parser()

    cfg = OmegaConf.load(hparams.cfg_path)

    pl.seed_everything(hparams.seed)

    # loading datasets
    loader = get_loader(cfg.data)

    # loading model from cfg
    model = instantiate_from_config(cfg.model)

    # configure learning rate
    model.learning_rate = cfg.model.base_learning_rate

    callbacks = []
    # val/loss_ema
    callbacks.append(ModelCheckpoint(monitor=cfg.model.params.monitor,
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))

    callbacks.append(ModelCheckpoint(every_n_train_steps=hparams.every_n_train_steps, save_top_k=-1,
                                     filename='{epoch}-{step}-{train/rec_loss:.2f}'))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    accelerator = 'ddp'

    trainer = pl.Trainer(
        # precision=hparams.precision,
        gpus=hparams.gpus,
        callbacks=callbacks,
        max_steps=hparams.max_steps,
        accelerator=accelerator,
        accumulate_grad_batches=1,
    )

    trainer.fit(model, loader[0], loader[1])


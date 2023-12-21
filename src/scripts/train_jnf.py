import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary
from models.exp_jnf import JNFExp
from models.models import FTJNF
from data.datamodule import HDF5DataModule
from utils.checkpoint import get_checkpointing_callback
from typing import Optional
from pytorch_lightning.cli import LightningCLI
import yaml

EXP_NAME='JNF'

def setup_logging(tb_log_dir: str, version_id: Optional[int]= None):
    """
    Set-up a Tensorboard logger.

    :param tb_log_dir: path to the log dir
    :param version_id: the version id (integer). Consecutive numbering is used if no number is given. 
    """

    if version_id is None:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False)

        # get current version id
        version_id = int((tb_logger.log_dir).split('_')[-1])
    else: 
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False, version=version_id)

    return tb_logger, version_id

def load_model(ckpt_file: str,
               _config):
    init_params = JNFExp.get_init_params(_config)
    model = JNFExp.load_from_checkpoint(ckpt_file, **init_params)
    model.to('cuda')
    return model

def get_trainer(devices, logger, max_epochs, gradient_clip_val, gradient_clip_algorithm, strategy, accelerator):
    return pl.Trainer(enable_model_summary=True,
                         logger=logger,
                         devices=devices,
                         log_every_n_steps=1,
                         max_epochs=max_epochs,
                         gradient_clip_val = gradient_clip_val,
                         gradient_clip_algorithm = gradient_clip_algorithm,
                         strategy = strategy,
                         accelerator = accelerator,
                         callbacks=[
                             #setup_checkpointing(),
                             ModelSummary(max_depth=2)
                                    ],

                         )

if __name__=="__main__":

    with open('config/jnf_config.yaml') as config_file: 
        config = yaml.safe_load(config_file)

    ## REPRODUCIBILITY
    pl.seed_everything(config.get('seed', 0), workers=True)

    ## LOGGING
    tb_logger, version = setup_logging(config['logging']['tb_log_dir'])

    ## DATA
    data_config = config['data']
    stft_length = data_config.get('stft_length_samples', 512)
    stft_shift = data_config.get('stft_shift_samples', 256)
    dm = HDF5DataModule(**data_config)

    ## CONFIGURE EXPERIMENT
    ckpt_file = config['training'].get('resume_ckpt', None)
    if not ckpt_file is None:
        exp = load_model(ckpt_file, config)
    else:
        model = FTJNF(**config['network'])
        exp = JNFExp(model=model,
                    stft_length=stft_length,
                    stft_shift=stft_shift,
                    **config['experiment'])

    ## TRAIN
    trainer = get_trainer(logger=tb_logger, **config['training'])
    trainer.fit(exp, dm)


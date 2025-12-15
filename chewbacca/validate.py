import os
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import submitit
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import lightning as pl

from lart import utils

log = utils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

@utils.task_wrapper
def validate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Validates the model using the best available checkpoint.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # Determine if we are using submitit
    try:
        env_information = submitit.JobEnvironment()
        log.info(f"Job ID: {int(env_information.job_id)}")
    except RuntimeError:
        log.info("Local job")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # look for latest checkpoint in logdir and load it if found
    checkpoint_path = cfg.get("ckpt_path")
    if not checkpoint_path:
        path_tmp = cfg.configs.storage_folder + "/checkpoints/last.ckpt"
        path_tmp_ema = cfg.configs.storage_folder + "/checkpoints/last-EMA.ckpt"
        try:
            list_ckpts = [f for f in os.listdir(cfg.configs.storage_folder + "/checkpoints/") if "last-v" in f and "EMA.ckpt" in f]
            if len(list_ckpts) > 0:
                list_ckpts = sorted(list_ckpts, key=lambda x: int(x.split("last-v")[1].split("-EMA.ckpt")[0]))
                path_tmp_ema = cfg.configs.storage_folder + "/checkpoints/" + list_ckpts[-1]
                log.info("Loading weights from last EMA ckpt " + path_tmp_ema)
        except:
            pass
        if os.path.exists(path_tmp_ema):
            checkpoint_path = path_tmp_ema
            log.info("Loading weights from last EMA ckpt " + checkpoint_path)
        elif os.path.exists(path_tmp):
            checkpoint_path = path_tmp
            log.info("Loading weights from last ckpt " + checkpoint_path)
        else:
            if(cfg.configs.weights_path is not None):

                if os.path.exists(cfg.configs.weights_path):

                    try:
                        av = torch.load(cfg.configs.weights_path, map_location=torch.device('cpu'))['state_dict']
                    except:
                        try:
                            av = torch.load(cfg.configs.weights_path, map_location=torch.device('cpu'))['model']
                            av = {"encoder."+k: v for k, v in av.items()}
                        except:
                            av = torch.load(cfg.configs.weights_path, map_location=torch.device('cpu'))
                            av = {k.replace("_forward_module.", ""): v for k, v in av.items()}

                    # if loading from torch.compile model
                    if(not cfg.configs.use_torch_compile):
                        av = {k.replace("._orig_mod", ""): v for k, v in av.items()}

                    if "remove-_forward" in cfg.configs.training_type:
                        av = {k.replace("_forward_module.", ""): v for k, v in av.items()}

                    if "remove-probe-layers" in cfg.configs.training_type:
                        av = {k: v for k, v in av.items() if "linear_layer" not in k}
                    if "interpolate-pos-emb" in cfg.configs.training_type:
                        from chewbacca.models.components.mae.models_mae_video import interpolate_pos_embed, interpolate_pos_embed2
                        if "encoder.decoder_pos_embed" in av:
                            a1 = av['encoder.decoder_pos_embed']
                            a2 = torch.randn_like(model.encoder.decoder_pos_embed)
                            a2[:, :a1.shape[1], :] = a1
                            av['encoder.decoder_pos_embed'] = a2


                    out = model.load_state_dict(av, strict=cfg.configs.load_strict)
                    log.info(out)
                    log.info("Loading weights from weights " + cfg.configs.weights_path)

            else:
                log.error("No valid checkpoint found for validation!")
                return {}, object_dict
    else:
        log.info("Loading weights from ckpt " + checkpoint_path)

    log.info("Starting validation!")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    log.info(f"Best ckpt path: {checkpoint_path}")

    val_metrics = trainer.callback_metrics

    return val_metrics, object_dict


@hydra.main(version_base="1.2", config_path="../configs", config_name="validate.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # validate the model
    metric_dict, _ = validate(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

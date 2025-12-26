from typing import Optional, Tuple

import hydra
import lightning as pl
import pyrootutils
from omegaconf import DictConfig

from vidgmae import utils
from vidgmae.utils.trainer_utils import (
    build_trainer,
    instantiate_components,
    log_job_environment,
    maybe_load_checkpoint,
)

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
    log_job_environment()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule, model, callbacks, logger = instantiate_components(cfg)
    trainer = build_trainer(cfg, callbacks, logger)

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

    checkpoint_path = maybe_load_checkpoint(cfg, model)
    if not checkpoint_path and not cfg.configs.weights_path:
        log.error("No valid checkpoint found for validation!")
        return {}, object_dict

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

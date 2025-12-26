import os
from copy import deepcopy
from typing import List, Optional, Tuple

import hydra
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from vidgmae import utils

log = utils.get_pylogger(__name__)


def log_job_environment() -> None:
    """Log submitit job id if available."""
    try:
        import submitit

        env_information = submitit.JobEnvironment()
        log.info(f"Job ID: {int(env_information.job_id)}")
    except Exception:
        log.info("Local job")


def instantiate_components(
    cfg: DictConfig,
) -> Tuple[LightningDataModule, LightningModule, List[Callback], List[Logger]]:
    """Hydra-instantiates the core Lightning pieces."""
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    return datamodule, model, callbacks, logger


def build_trainer(
    cfg: DictConfig, callbacks: List[Callback], logger: List[Logger]
) -> Trainer:
    """Constructs the Trainer, handling the FSDP special case."""
    if cfg.trainer.get("strategy") == "fsdp":
        strategy = _build_fsdp_strategy(cfg)
        trainer_kwargs = _trainer_kwargs(cfg.trainer)
        return Trainer(
            callbacks=callbacks, logger=logger, strategy=strategy, **trainer_kwargs
        )

    return hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)


def _trainer_kwargs(trainer_cfg: DictConfig) -> dict:
    """Convert a Trainer DictConfig into kwargs the class accepts."""
    container = OmegaConf.to_container(deepcopy(trainer_cfg), resolve=True)
    return {k: v for k, v in container.items() if k not in {"_target_", "strategy"}}


def _build_fsdp_strategy(cfg: DictConfig):
    """Build a lean FSDP strategy; keeps Trainer args declarative."""
    from functools import partial

    from lightning.pytorch.strategies import FSDPStrategy
    from timm.models.vision_transformer import Block
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    use_orig = "orig" in cfg.configs.training_type
    return FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=Block,
        limit_all_gathers=True,
        use_orig_params=use_orig,
    )


def maybe_load_checkpoint(cfg: DictConfig, model: LightningModule) -> Optional[str]:
    """Resolve a checkpoint path or load loose weights onto the model."""
    checkpoint_path = cfg.get("ckpt_path")
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            log.info(f"Loading weights from ckpt {checkpoint_path}")
            return checkpoint_path
        log.warning(f"Provided ckpt_path does not exist, skipping: {checkpoint_path}")

    checkpoint_path = _latest_checkpoint(cfg.configs.storage_folder)
    if checkpoint_path:
        log.info(f"Loading weights from last ckpt {checkpoint_path}")
        return checkpoint_path

    if cfg.configs.weights_path:
        _load_weights(cfg, model)
    else:
        log.warning(
            "No checkpoint or weights_path found; using randomly initialized weights."
        )
    return None


def _latest_checkpoint(storage_folder: str) -> Optional[str]:
    ckpt_dir = os.path.join(storage_folder, "checkpoints")
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    last_ema = os.path.join(ckpt_dir, "last-EMA.ckpt")
    try:
        ema_candidates = [
            f for f in os.listdir(ckpt_dir) if "last-v" in f and "EMA.ckpt" in f
        ]
    except FileNotFoundError:
        ema_candidates = []

    if ema_candidates:
        ema_candidates = sorted(ema_candidates, key=_ema_version_key)
        last_ema = os.path.join(ckpt_dir, ema_candidates[-1])

    if os.path.exists(last_ema):
        return last_ema
    if os.path.exists(last_ckpt):
        return last_ckpt
    return None


def _ema_version_key(filename: str) -> int:
    try:
        return int(filename.split("last-v")[1].split("-EMA.ckpt")[0])
    except Exception:
        return -1


def _load_weights(cfg: DictConfig, model: LightningModule) -> None:
    weights_path = cfg.configs.weights_path
    if not os.path.exists(weights_path):
        log.error(f"weights_path provided but file is missing: {weights_path}")
        return

    log.info(f"Loading weights from {weights_path}")
    state_dict = _load_state_dict(weights_path)

    if not cfg.configs.use_torch_compile:
        state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}

    if "remove-_forward" in cfg.configs.training_type:
        state_dict = {
            k.replace("_forward_module.", ""): v for k, v in state_dict.items()
        }

    if "remove-probe-layers" in cfg.configs.training_type:
        state_dict = {k: v for k, v in state_dict.items() if "linear_layer" not in k}

    if "interpolate-pos-emb" in cfg.configs.training_type:
        state_dict = _interpolate_pos_embed(state_dict, model)

    out = model.load_state_dict(state_dict, strict=cfg.configs.load_strict)
    log.info(out)


def _load_state_dict(weights_path: str) -> dict:
    """Handle the variants of stored keys we have seen."""
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
    for key in ("state_dict", "model"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            if key == "model":
                state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}
            return state_dict
    return {k.replace("_forward_module.", ""): v for k, v in checkpoint.items()}


def _interpolate_pos_embed(state_dict: dict, model: LightningModule) -> dict:
    """Pad/resize decoder pos embed if checkpoint was trained at a different length."""
    if "encoder.decoder_pos_embed" not in state_dict:
        return state_dict
    a1 = state_dict["encoder.decoder_pos_embed"]
    a2 = torch.randn_like(model.encoder.decoder_pos_embed)
    a2[:, : a1.shape[1], :] = a1
    state_dict["encoder.decoder_pos_embed"] = a2
    return state_dict

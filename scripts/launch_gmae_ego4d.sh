#!/bin/bash

# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge2 \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_ego4d_v3_256 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.inference.testing=True \
# configs.inference.depth=False \
# configs.inference.correspondences=False \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=112 \
# configs.lr=1e-3 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=False,True \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000 \
# trainer.limit_val_batches=1000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.mask_ratio=0.75,0.95 \
# configs.training_type="ego4d_gaussian_save-images-z_no-mask_random-frames"











python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_scavenge2 \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vit_base_ego4d_v4_256 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=8 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.lr=1e-3,1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=20 \
trainer.max_epochs=800 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=False,True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000 \
trainer.limit_val_batches=1000 \
callbacks.model_checkpoint.every_n_epochs=1 \
configs.mask_ratio=0.75,0.95 \
configs.training_type="ego4d_gaussian_save-images-z_no-mask_random-frames"
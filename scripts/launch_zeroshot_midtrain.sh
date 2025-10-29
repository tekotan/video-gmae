# !/bin/bash
# hydra/launcher=submitit_slurm \
# launcher=slurm_em \

python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_base_zeroshot_midtrain \
trainer=ddp_unused \
trainer.devices=4 \
trainer.num_nodes=1 \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=224 \
configs.lr=5e-5 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=4 \
configs.test_batch_size=4 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=20 \
trainer.max_epochs=100 \
configs.seq_length=16 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.vocab_size=256 \
configs.num_classes=400 \
configs.dataset_type="video" \
trainer.limit_train_batches=50000000 \
trainer.limit_val_batches=5000000 \
callbacks.model_checkpoint.every_n_epochs=1 \
configs.weights_path="/home/jathu/gmae_logs/vgmae_base_k400_test1/3/checkpoints/last.ckpt" \
configs.training_type="zeroshot_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.load_strict=False \
configs.mask_ratio=0.90

# # !/bin/bash
# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_em \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vgmae_base_kubric_midtrain \
# trainer=ddp_unused \
# trainer.devices=4 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=5e-5 \
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
# configs.seq_length=16 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.vocab_size=256 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=50000000 \
# trainer.limit_val_batches=5000000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.weights_path="/home/jathu/gmae_logs/vgmae_base_k400_test1/3/checkpoints/last.ckpt" \
# configs.training_type="kubric_gaussian_save-images-z_no-mask_random-frames-attn" \
# configs.load_strict=False \
# configs.mask_ratio=0.90
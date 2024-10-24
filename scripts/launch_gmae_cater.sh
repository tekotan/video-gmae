#!/bin/bash
python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vit_base_cater_v1_test8 \
trainer=ddp_unused \
trainer.devices=4 \
trainer.num_nodes=1 \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=112 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=4 \
configs.test_batch_size=1 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=20 \
trainer.max_epochs=800 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.00 \
configs.random_frames=True \
configs.sample_rate=1 \
configs.num_classes=200 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=500 \
trainer.limit_val_batches=500 \
callbacks.model_checkpoint.every_n_epochs=1 \
configs.training_type="cater_gaussian_save-images-z_no-mask_random-frames"

# #!/bin/bash
# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_cater_v0_sanitycheck5 \
# trainer=ddp_unused \
# trainer.devices=1 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=112 \
# configs.lr=1e-5 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=1 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=405 \
# configs.seq_length=8 \
# configs.sample_rate=1 \
# configs.num_classes=200 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=500 \
# trainer.limit_val_batches=500 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="cater_gaussian_save-images-z_no-mask"
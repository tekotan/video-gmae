# !/bin/bash
python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_em \
model._target_="chewbacca.models.finetune.FinetuneLitModule" \
datamodule._target_="chewbacca.datamodules.finetune_datamodule.FinetuneDataModule" \
task_name=vit_base_ucf101_finetune_tracking_v3 \
trainer=ddp_unused \
trainer.devices=1 \
trainer.num_nodes=1 \
configs.task="finetune" \
configs.model_name="finetune_vit_base_patch16" \
configs.input_size=112 \
configs.lr=5e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=4 \
configs.test_batch_size=4 \
configs.train_num_workers=1 \
configs.test_num_workers=1 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=20 \
trainer.max_epochs=800 \
configs.seq_length=32 \
configs.num_classes=200 \
configs.dataset_type="video" \
trainer.limit_train_batches=5000 \
trainer.limit_val_batches=1000 \
callbacks.model_checkpoint.every_n_epochs=1 \
configs.training_type="train_ucf101_point-tracking_save-images_no-mask_random-frames" \
configs.weights_path="/home/jathu/3D/Chewbacca_test/logs/logs/GMAE_ucf101_test1/3/checkpoints/epoch_1561.ckpt" \
configs.load_strict=False \
configs.mask_ratio=0.5 \
configs.finetune_params.reuse_decoder=False \
configs.finetune_params.num_fourier_features=0,16,32




# # #!/bin/bash
# python chewbacca/validate.py -m \
# --config-name gmae_ema.yaml \
# model._target_="chewbacca.models.finetune.FinetuneLitModule" \
# datamodule._target_="chewbacca.datamodules.finetune_datamodule.FinetuneDataModule" \
# task_name=vit_base_ucf101_finetune_tracking \
# trainer=ddp_unused \
# trainer.devices=1 \
# trainer.num_nodes=1 \
# configs.task="finetune" \
# configs.model_name="finetune_vit_base_patch16" \
# configs.input_size=112 \
# configs.lr=1e-3 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=4 \
# configs.test_batch_size=4 \
# configs.train_num_workers=1 \
# configs.test_num_workers=1 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.num_classes=200 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=5000 \
# trainer.limit_val_batches=1000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="eval_ucf101_point-tracking_save-images_no-mask_random-frames" \
# configs.load_strict=False \
# configs.mask_ratio=0.5 \
# configs.finetune_params.reuse_decoder=True \
# configs.finetune_params.num_fourier_features=0 \
# configs.inference.testing=True
# !/bin/bash
python chewbacca/validate.py -m \
--config-name gmae_ema.yaml \
model._target_="chewbacca.models.finetune.FinetuneLitModule" \
datamodule._target_="chewbacca.datamodules.finetune_datamodule.FinetuneDataModule" \
task_name=vgmae_large_seq16_finetune_point_v1 \
trainer=ddp_unused \
trainer.devices=1 \
trainer.num_nodes=1 \
configs.task="finetune" \
configs.model_name="finetune_vit_large_patch16" \
configs.input_size=224 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=8 \
configs.test_batch_size=8 \
configs.train_num_workers=1 \
configs.test_num_workers=1 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=20 \
trainer.max_epochs=200 \
configs.seq_length=16 \
configs.num_classes=400 \
configs.dataset_type="video" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=1 \
configs.training_type="eval-davis_ucf101_point-tracking_save-images_no-mask_interpolate-pos-emb" \
configs.weights_path="./checkpoints/finetune_checkpoint.ckpt" \
configs.load_strict=False \
configs.mask_ratio=0.0 \
configs.inference.testing=True \
configs.inference.context_length=1 \
configs.inference.save_predictions=True


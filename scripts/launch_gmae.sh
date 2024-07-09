# #!/bin/bash


python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
task_name=vit_base_imagenet_test1 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.task_2="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=224 \
configs.patch_size=8 \
configs.dataset_type="imagenet" \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=16 \
configs.test_batch_size=16 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=20 \
trainer.max_epochs=400 \
configs.num_classes=1000 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="imagenet_gaussian_save-images-z_no-mask_loss-masked" \
configs.scale_factor=1.0 \
configs.scale_vocab=1 \
configs.mask_ratio=0.75 \
configs.solver="AdamW" \
# trainer.limit_train_batches=100 \
# trainer.limit_val_batches=100 \




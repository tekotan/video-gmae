# #!/bin/bash


python chewbacca/train.py -m \
--config-name gmae.yaml \
model._target_="chewbacca.models.gdit.GDITLitModule" \
datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
task_name=vit_base_imagenet_dit \
configs.lr=1e-2 \
trainer=ddp_unused \
trainer.devices=1 \
trainer.num_nodes=1 \
configs.task="pretrain" \
configs.model_name="dit-a" \
configs.input_size=224 \
configs.dataset_type="imagenet" \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=16 \
configs.test_batch_size=16 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=400 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.num_classes=1000 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="save-images_overfit" \
configs.vocab_size=256 \
configs.scale_factor=1.0 \
configs.scale_vocab=1 \
configs.mask_ratio=0.75 \
configs.solver="AdamW" \
# trainer.limit_train_batches=10 \
# trainer.limit_val_batches=10 \




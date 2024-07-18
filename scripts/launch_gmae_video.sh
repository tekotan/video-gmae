# #!/bin/bash



# # Gaussian video+image model
# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_k400_test1 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-3 \
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
# trainer.max_epochs=400 \
# configs.seq_length=12 \
# configs.sample_rate=8 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=300 \
# trainer.limit_val_batches=300 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="k400_gaussian_save-images-z_no-mask" \









# # Gaussian video+image model
# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_dev \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_cater_test1 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-3 \
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
# trainer.max_epochs=400 \
# configs.seq_length=12 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=300 \
# trainer.limit_val_batches=300 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="cater_gaussian_save-images-z_no-mask" \









# # Gaussian video+image model
# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_cater_test2 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-3 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=4 \
# configs.test_batch_size=1 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=1000 \
# configs.seq_length=12 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000 \
# trainer.limit_val_batches=300 \
# configs.vocab_size=512 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="cater_gaussian_save-images-z_no-mask" \











# # Gaussian video+image model
# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_cater_test3 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.lr=1e-3 \
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
# trainer.max_epochs=1000 \
# configs.seq_length=12 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000 \
# trainer.limit_val_batches=300 \
# configs.vocab_size=512 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="cater_gaussian_save-images-z_no-mask" \
# trainer.strategy="fsdp" \
# +trainer.gradient_clip_algorithm="value" \












# # Gaussian video+image model
# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_k400_test4 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.lr=1e-3 \
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
# trainer.max_epochs=1000 \
# configs.seq_length=12 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000 \
# trainer.limit_val_batches=300 \
# configs.vocab_size=512 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="k400_gaussian_save-images-z_no-mask" \
# trainer.strategy="fsdp" \
# +trainer.gradient_clip_algorithm="value" \













# Gaussian video+image model
python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_scavenge \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vit_base_k400_test5 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.lr=4e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=1 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=20 \
trainer.max_epochs=1000 \
configs.seq_length=12 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000 \
trainer.limit_val_batches=300 \
configs.vocab_size=512 \
callbacks.model_checkpoint.every_n_epochs=1 \
configs.training_type="k400_gaussian_save-images-z_no-mask" \
trainer.strategy="fsdp" \
+trainer.gradient_clip_algorithm="value" \



# !/bin/bash



# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_a_pretraining_test1 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=4 \
# configs.train_num_workers=16 \
# configs.test_num_workers=16 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.num_classes=174 \
# configs.vocab_size=256 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=50000000 \
# trainer.limit_val_batches=5000000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="ucf101_gaussian_save-images-z_no-mask_random-frames" \
# configs.load_strict=False \
# configs.mask_ratio=0.95










# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_a_pretraining_test2 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-5 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=4 \
# configs.train_num_workers=16 \
# configs.test_num_workers=16 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.num_classes=174 \
# configs.vocab_size=256 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=50000000 \
# trainer.limit_val_batches=5000000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="ucf101_gaussian_save-images-z_no-mask_random-frames" \
# configs.load_strict=False \
# configs.mask_ratio=0.95












# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_a_pretraining_test3 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-5 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=4 \
# configs.train_num_workers=16 \
# configs.test_num_workers=16 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.num_classes=174 \
# configs.vocab_size=256 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=50000000 \
# trainer.limit_val_batches=5000000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="ucf101_gaussian_save-images-z_no-mask_random-frames" \
# configs.load_strict=False \
# configs.mask_ratio=0.95

















# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_base_a_pretraining_test4 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.lr=1e-5 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=4 \
# configs.train_num_workers=16 \
# configs.test_num_workers=16 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.num_classes=174 \
# configs.vocab_size=256 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=50000000 \
# trainer.limit_val_batches=5000000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="ucf101_gaussian_save-images-z_no-mask_random-frames" \
# configs.load_strict=False \
# configs.mask_ratio=0.95












python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=a_vit_base_pretraining_test0 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=8 \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=224 \
configs.lr=1e-5 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=4 \
configs.train_num_workers=16 \
configs.test_num_workers=16 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=800 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.num_classes=174 \
configs.vocab_size=256 \
configs.dataset_type="video" \
trainer.limit_train_batches=50000000 \
trainer.limit_val_batches=5000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="ucf101_gaussian_save-images-z_no-mask_random-frames" \
configs.load_strict=False \
configs.mask_ratio=0.95









# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vit_large_a_pretraining_test1 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=4 \
# configs.train_num_workers=16 \
# configs.test_num_workers=16 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=20 \
# trainer.max_epochs=800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.num_classes=174 \
# configs.vocab_size=256 \
# configs.dataset_type="video" \
# trainer.limit_train_batches=50000000 \
# trainer.limit_val_batches=5000000 \
# callbacks.model_checkpoint.every_n_epochs=1 \
# configs.training_type="ucf101_gaussian_save-images-z_no-mask_random-frames" \
# configs.load_strict=False \
# configs.mask_ratio=0.95

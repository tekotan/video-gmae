#!/bin/bash

# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=b_vit_base_pretraining_test0 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.inference.testing=True \
# configs.inference.depth=False \
# configs.inference.correspondences=False \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=256 \
# configs.lr=1e-4,1e-3 \
# configs.weight_decay=5e-2,5e-4 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=8 \
# configs.test_batch_size=8 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=400,800 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True,False \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000000 \
# trainer.limit_val_batches=1000000 \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames"











# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=b_vit_base_pretraining_test1 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.inference.testing=True \
# configs.inference.depth=False \
# configs.inference.correspondences=False \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=256 \
# configs.lr=4e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=400 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000000 \
# trainer.limit_val_batches=1000000 \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames"














# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=b_vit_base_pretraining_test2 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.inference.testing=True \
# configs.inference.depth=False \
# configs.inference.correspondences=False \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=256 \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=400 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000000 \
# trainer.limit_val_batches=1000000 \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames"

















# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=b_vit_base_pretraining_test3 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.inference.testing=True \
# configs.inference.depth=False \
# configs.inference.correspondences=False \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=256 \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=400 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000000 \
# trainer.limit_val_batches=1000000 \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames"




















# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=b_vit_base_pretraining_test4 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.inference.testing=True \
# configs.inference.depth=False \
# configs.inference.correspondences=False \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=256 \
# configs.lr=4e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# configs.warmup_steps=2000 \
# trainer.max_epochs=400 \
# configs.seq_length=32 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=True \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.num_classes=400 \
# configs.load_strict=False \
# configs.dataset_type="video" \
# trainer.limit_train_batches=1000000 \
# trainer.limit_val_batches=1000000 \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames" \
# configs.mask_ratio=0.95


















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test5 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=4e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames" \
configs.mask_ratio=0.95













python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test6 \
trainer=ddp_unused \
trainer.devices=1 \
trainer.num_nodes=1 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test7_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95


















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test7_32k7 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95







python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test8_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \


















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test8_32k7 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \













python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test9_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames_attn_loss-masked" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \










python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_base_pretraining_test10_32k6 \
trainer=ddp_unused \
trainer.devices=1 \
trainer.num_nodes=1 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn_full-loop" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \







































































python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test1_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \

















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test2_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=4e-4,1e-3 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \
















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test3_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=4e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=4000 \
trainer.max_epochs=1600 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \
















python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test4_32k6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=4000 \
trainer.max_epochs=1600 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video-vjepa" \
configs.video_source="video-vjepa" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400-vjepa_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95 \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/b_vit_base_pretraining_test6/0/checkpoints/last.ckpt" \










python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=400 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95






# export LD_LIBRARY_PATH='/private/home/jathushan/anaconda3/envs/chewbacca_gs3/lib':$LD_LIBRARY_PATH

python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test7 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=1600 \
configs.seq_length=32 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=True \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95














python chewbacca/train.py -m \
--config-name gmae_ema.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=b_vit_large_pretraining_test8 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.inference.testing=True \
configs.inference.depth=False \
configs.inference.correspondences=False \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=256 \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.scheduler="cosine_step" \
configs.lr_interval="step" \
configs.warmup_steps=2000 \
trainer.max_epochs=1600 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.num_classes=400 \
configs.load_strict=False \
configs.dataset_type="video" \
trainer.limit_train_batches=1000000 \
trainer.limit_val_batches=1000000 \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn" \
configs.mask_ratio=0.95
















# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae2.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
# task_name=vgmae_large_k400_test1 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.dataset_type="video" \
# configs.lr=1e-4 \
# configs.weight_decay=0.0002 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=10 \
# trainer.max_epochs=90 \
# configs.num_classes=400 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
# configs.vocab_size=256 \
# configs.seq_length=8 \
# configs.deltas_reg_weight=0.0 \
# configs.random_frames=False \
# configs.rgb_deltas=True \
# configs.mean_deltas=True \
# configs.scale_vocab=1 \
# configs.sample_rate=1 \
# configs.mask_ratio=0.95 \
# configs.solver="AdamW" \
# configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \











python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test1a \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=2,4 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \







python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test1b \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=4 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \











python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test2 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=40 \
trainer.max_epochs=400 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \

















python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test3 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=80 \
trainer.max_epochs=1600 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \














python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test4 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=16 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \



























python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test5 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt","/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last_.ckpt" \















python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test6 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=24 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \

















python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_large_k400_test7 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=8 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True,False \
configs.mean_deltas=True,False \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_large_imagenet_test1/0/checkpoints/last.ckpt" \






















python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae2.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.video_datamodule.VideoDataModule" \
task_name=vgmae_base_k400_test1 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=224 \
configs.dataset_type="video" \
configs.lr=1e-4 \
configs.weight_decay=0.0002 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=2 \
configs.test_batch_size=2 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=10 \
trainer.max_epochs=90 \
configs.num_classes=400 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="k400_gaussian_save-images-z_no-mask_random-frames-attn_remove-probe-layers_interpolate-pos-emb" \
configs.vocab_size=256 \
configs.seq_length=2,4,8,16,24 \
configs.deltas_reg_weight=0.0 \
configs.random_frames=False \
configs.rgb_deltas=True \
configs.mean_deltas=True \
configs.scale_vocab=1 \
configs.sample_rate=1 \
configs.mask_ratio=0.95 \
configs.solver="AdamW" \
configs.weights_path="/private/home/jathushan/3D/video_gmae/logs/gmae_base_imagenet_test3/1/checkpoints/last.ckpt" \



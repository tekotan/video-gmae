# #!/bin/bash




python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
task_name=gmae_base_imagenet_test1 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=224 \
configs.dataset_type="imagenet" \
configs.lr=1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=4 \
configs.test_batch_size=4 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=20 \
trainer.max_epochs=400 \
configs.num_classes=1000 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
configs.vocab_size=256 \
configs.scale_factor=1.0 \
configs.scale_vocab=1 \
configs.mask_ratio=0.75 \
configs.solver="AdamW" \
# trainer.limit_train_batches=100 \
# trainer.limit_val_batches=100 \










# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm_x \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_base_imagenet_test2 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=4e-5,1e-5,4e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=400 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \

















python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
task_name=gmae_base_imagenet_test3 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_base_patch16" \
configs.input_size=224 \
configs.dataset_type="imagenet" \
configs.lr=4e-5,6e-5,2e-5 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=16 \
configs.test_batch_size=16 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=20 \
trainer.max_epochs=1600 \
configs.num_classes=1000 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
configs.vocab_size=256 \
configs.scale_factor=1.0 \
configs.scale_vocab=1 \
configs.mask_ratio=0.75 \
configs.solver="AdamW" \
# trainer.limit_train_batches=100 \
# trainer.limit_val_batches=100 \

















python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm_x \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
task_name=gmae_large_imagenet_test1 \
trainer=ddp_unused \
trainer.devices=8 \
trainer.num_nodes=32 \
configs.task="pretrain" \
configs.model_name="mae_vit_large_patch16" \
configs.input_size=224 \
configs.dataset_type="imagenet" \
configs.lr=4e-5,6e-5,2e-5,1e-4 \
configs.weight_decay=5e-2 \
trainer.accumulate_grad_batches=1 \
configs.train_batch_size=16 \
configs.test_batch_size=16 \
configs.train_num_workers=8 \
configs.test_num_workers=8 \
trainer.gradient_clip_val=2.0 \
configs.warmup_steps=20 \
trainer.max_epochs=1600 \
configs.num_classes=1000 \
configs.load_strict=False \
callbacks.model_checkpoint.every_n_epochs=10 \
configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
configs.vocab_size=256 \
configs.scale_factor=1.0 \
configs.scale_vocab=1 \
configs.mask_ratio=0.75 \
configs.solver="AdamW" \
# trainer.limit_train_batches=100 \
# trainer.limit_val_batches=100 \








# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_base_imagenet_ft1 \
# trainer=ddp_unused \
# configs.task="pretrain" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.vocab_size=256 \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# configs.lr=1e-5,4e-5 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10,20 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/gmae_base_imagenet/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit_full-finetuning_remove-probe-layers" \
# configs.solver="AdamW" \
# callbacks.ema.decay=0.999,0.99 \
# configs.drop_path=0.0,0.1,0.2 \









# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_base_imagenet_test1 \
# trainer=ddp_unused \
# trainer.devices=1 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=400 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/gmae_base_imagenet/0/checkpoints/last.ckpt" \
# trainer.limit_val_batches=100000 \
# train=False \
# # trainer.limit_train_batches=100 \



























# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=1600 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \








# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_test2 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# trainer.precision=16 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=2e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=1600 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \






# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_test3 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# trainer.precision=16 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=2e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=8 \
# configs.test_batch_size=8 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=1000 \
# trainer.max_epochs=10 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked_interpolate-pos-emb" \
# configs.vocab_size=1024 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# configs.weights_path="/private/home/jathushan/3D/HumanMAE/logs/rsc_ckpts/ckpts/rsc_vit_large_imagenet_test3x/0/checkpoints/last.ckpt" \
# configs.scheduler="cosine_step" \
# configs.lr_interval="step" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \








# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_test4 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=1e-3 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=80 \
# trainer.max_epochs=1600 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \










# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_test2 \
# trainer=ddp_unused \
# trainer.devices=1 \
# trainer.num_nodes=1 \
# configs.task="pretrain" \
# trainer.precision=16 \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=1 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# trainer.limit_train_batches=100 \
# trainer.limit_val_batches=100 \





# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_test6 \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=8 \
# configs.task="pretrain" \
# trainer.precision=16 \
# configs.model_name="mae_vit_large_patch16" \
# configs.input_size=256 \
# configs.dataset_type="imagenet" \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=2 \
# configs.test_batch_size=2 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=10000 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=1024 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# configs.weights_path="/private/home/jathushan/3D/HumanMAE/logs/rsc_ckpts/ckpts/rsc_vit_large_imagenet_test3x/0/checkpoints/last.ckpt" \
# trainer.limit_train_batches=1000 \
# trainer.limit_val_batches=1000 \








# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_ft1 \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# configs.lr=1e-3,1e-4,4e-4,1e-5 \
# configs.weight_decay=0,0.02,0.0002 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_large_patch16" \
# configs.input_size=256 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/HumanMAE/logs/rsc_ckpts/ckpts/rsc_vit_large_imagenet_test3x/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit_full-finetuning_remove-probe-layers" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \









# python chewbacca/train.py -m \
# --config-name gmae_ema.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=gmae_large_imagenet_ft2 \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# configs.lr=1e-5,4e-5 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10,20 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_large_patch16" \
# configs.input_size=256 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/HumanMAE/logs/rsc_ckpts/ckpts/rsc_vit_large_imagenet_test3x/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit_full-finetuning_remove-probe-layers" \
# configs.solver="AdamW" \
# callbacks.ema.decay=0.999,0.99 \
# configs.drop_path=0.0,0.1,0.2 \











































# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=vit_base_imagenet \
# trainer=ddp_unused \
# trainer.devices=8 \
# trainer.num_nodes=32 \
# configs.task="pretrain" \
# configs.model_name="mae_vit_base_patch16" \
# configs.input_size=224 \
# configs.dataset_type="imagenet" \
# configs.lr=1e-4 \
# configs.weight_decay=5e-2 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=16 \
# configs.test_batch_size=16 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_steps=20 \
# trainer.max_epochs=400 \
# configs.num_classes=1000 \
# configs.load_strict=False \
# callbacks.model_checkpoint.every_n_epochs=10 \
# configs.training_type="mae_imagenet_gaussian_save-images-z_no-mask_loss-masked" \
# configs.vocab_size=256 \
# configs.scale_factor=1.0 \
# configs.scale_vocab=1 \
# configs.mask_ratio=0.75 \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \










# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp1 \
# configs.lr=0.1 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit" \
# configs.solver="LARS" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \









# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp2 \
# configs.lr=1e-3,1e-2,1e-1 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit" \
# configs.solver="LAMB" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \








# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp3 \
# configs.lr=1e-3,1e-2,1e-1 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=128 \
# configs.test_batch_size=128 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit" \
# configs.solver="LAMB" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \








# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp4 \
# configs.lr=1e-3,1e-2,1e-1 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=128 \
# configs.test_batch_size=128 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="pretrain" \
# configs.training_type="imagenet_vit" \
# configs.solver="LAMB" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \












# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp5 \
# configs.lr=1e-4,2e-4,4e-4 \
# configs.weight_decay=0,0.2,0.02,0.002,0.0002 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=64 \
# configs.test_batch_size=64 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="pretrain" \
# configs.training_type="imagenet_vit" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \















# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp6 \
# configs.lr=1e-4,2e-4,4e-4 \
# configs.weight_decay=0,0.2,0.02,0.002,0.0002 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=64 \
# configs.test_batch_size=64 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="pretrain" \
# configs.training_type="imagenet_vit_remove-probe-layers" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \







# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# task_name=vit_base_imagenet_test1_lp7 \
# configs.lr=1e-4,2e-4,4e-4 \
# configs.weight_decay=0,0.2,0.02,0.002,0.0002 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=64 \
# configs.test_batch_size=64 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="pretrain" \
# configs.training_type="imagenet_vit_remove-probe-layers" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \




















































































# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=vit_base_imagenet_test1_ft1 \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# configs.lr=1e-3,1e-4,4e-4,1e-5 \
# configs.weight_decay=0 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit_full-finetuning" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \





# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=vit_base_imagenet_test1_ft2 \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# configs.lr=1e-3,1e-4,4e-4,1e-5 \
# configs.weight_decay=0,0.02,0.0002 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit_full-finetuning_remove-probe-layers" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \









# python chewbacca/train.py -m \
# --config-name gmae.yaml \
# trainer=ddp_unused \
# hydra/launcher=submitit_slurm \
# launcher=slurm_scavenge \
# model._target_="chewbacca.models.gmae.GMAELitModule" \
# datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
# task_name=vit_base_imagenet_test1_ft3 \
# trainer.devices=8 \
# trainer.num_nodes=4 \
# configs.lr=1e-3,1e-4,4e-4,1e-5 \
# configs.weight_decay=0,0.02,0.0002 \
# configs.mask_ratio=0.75 \
# trainer.accumulate_grad_batches=1 \
# configs.train_batch_size=32 \
# configs.test_batch_size=32 \
# configs.train_num_workers=8 \
# configs.test_num_workers=8 \
# trainer.gradient_clip_val=2.0 \
# configs.warmup_epochs=10 \
# trainer.max_epochs=90 \
# callbacks.rich_progress_bar.refresh_rate=0 \
# configs.dataset_type="imagenet" \
# configs.model_name="vit_base_patch16" \
# configs.input_size=224 \
# configs.load_strict=False \
# configs.weights_path="/private/home/jathushan/3D/Chewbacca_test/logs/vit_base_imagenet_test1/0/checkpoints/last.ckpt" \
# configs.task="finetune" \
# configs.training_type="imagenet_vit_full-finetuning_remove-probe-layers" \
# configs.solver="AdamW" \
# # trainer.limit_train_batches=100 \
# # trainer.limit_val_batches=100 \



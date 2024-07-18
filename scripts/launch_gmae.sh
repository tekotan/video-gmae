# #!/bin/bash


python chewbacca/train.py -m \
--config-name gmae.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm \
model._target_="chewbacca.models.gmae.GMAELitModule" \
datamodule._target_="chewbacca.datamodules.image_datamodule.ImageDataModule" \
task_name=vit_base_imagenet \
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



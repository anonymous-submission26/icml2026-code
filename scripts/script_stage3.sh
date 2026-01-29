#!/bin/bash

# This script is used for stage 2 training of Orion-MSP

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------


# Loading prior data from disk and training
torchrun --standalone --nproc_per_node=8 /path/to/orion_msp/train/run.py \
            --wandb_log True \
            --wandb_project OrionMSP_v1.5 \
            --wandb_name Stage3 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 2000 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 2e-5 \
            --scheduler polynomial_decay_warmup \
            --warmup_proportion 0 \
            --poly_decay_lr_end 5e-6 \
            --poly_decay_power 2.0 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --min_seq_len 30000 \
            --max_seq_len 60000 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --batch_size_per_gp 1 \
            --prior_device cpu \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --feature_pos_emb subspace \
            --row_num_blocks 9 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_scales 1,4,16 \
            --features_per_group 2 \
            --scale_combine_method enhanced_attention \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --save_temp_every 25 \
            --save_perm_every 200 \
            --row_window 8 \
            --row_num_random 2 \
            --row_num_global 2 \
            --row_group_mode pma \
            --perc_num_latents 32 \
            --perc_layers 2 \
            --num_memory_heads 4 \
            --use_memory_gating True \
            --num_thinking_tokens 2 \
            --checkpoint_dir /my/stage3/checkpoint/dir \
            --checkpoint_path /my/stage2/checkpoint/dir/step-{last}.ckpt \


# Loading prior data from disk and training
torchrun --standalone --nproc_per_node=8 /path/to/orion_msp/train/run.py \
            --wandb_log True \
            --wandb_project OrionMSP_v1.5.2 \
            --wandb_name Stage3 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 3000 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 5e-5 \
            --scheduler polynomial_decay_warmup \
            --warmup_proportion 0 \
            --poly_decay_lr_end 5e-6 \
            --poly_decay_power 2.0 \
            --gradient_clipping 1.0 \
            --prior_dir /path/to/prior/stage3/ \
            --prior_type mix_scm \
            --load_prior_start 0 \
            --batch_size_per_gp 2 \
            --delete_after_load False \
            --prior_device cpu \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --feature_pos_emb subspace \
            --row_num_blocks 9 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_scales 1,4,16 \
            --features_per_group 2 \
            --scale_combine_method enhanced_attention \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --save_temp_every 25 \
            --save_perm_every 200 \
            --row_window 8 \
            --row_num_random 2 \
            --row_num_global 2 \
            --row_group_mode pma \
            --perc_num_latents 32 \
            --perc_layers 2 \
            --num_memory_heads 4 \
            --use_memory_gating True \
            --num_thinking_tokens 4 \
            --checkpoint_dir /my/stage3/checkpoint/dir \
            --checkpoint_path /my/stage2/checkpoint/dir/step-{last}.ckpt \

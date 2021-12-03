#!/bin/bash

SCRIPT_DIR=.
MODEL_DIR=/mnt/disks/data-1/models/training_indobert_large_unfreeze

IMAGE_ENCODER="openai/clip-vit-base-patch32"
TEXT_ENCODER="indobenchmark/indobert-large-p2"

python ${SCRIPT_DIR}/run_hybrid_clip.py \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir \
    --tokenizer_name=${TEXT_ENCODER} \
    --train_file="../data/train_dataset_v6.json" \
    --validation_file="../data/val_dataset_v6.json" \
    --do_train --do_eval \
    --num_train_epochs="20" --max_seq_length 96 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="0.00001" --warmup_ratio 0.1 --weight_decay 0.0 \
    --preprocessing_num_workers 16 \
    --exp_name training_v3_unfreeze \
    --text_model_name_or_path=${TEXT_ENCODER} \
    --vision_model_name_or_path=${IMAGE_ENCODER} \
    --eval_steps 500 \
    --logging_steps 100 \
    --save_steps 500 \
    --save_total_limit 5 \
    --log_wandb \
    --run_from_checkpoint="/mnt/disks/data-1/models/training_v3/ckpt-39999" # edit
    #--freeze_backbones
    #--push_to_hub
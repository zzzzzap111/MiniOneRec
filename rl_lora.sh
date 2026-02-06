#!/bin/bash
# MiniOneRec RL 训练脚本（支持 LoRA）

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export CUDA_VISIBLE_DEVICES=0   # 单卡训练

# 配置参数
SFT_MODEL_PATH="./output/sft_lora_qwen25_3b/final_checkpoint"  # SFT 模型路径
USE_LORA=True                                                   # 是否使用 LoRA
LORA_R=64                                                       # LoRA rank（必须与 SFT 一致）
LORA_ALPHA=128                                                  # LoRA alpha（必须与 SFT 一致）
LORA_DROPOUT=0.05                                               # LoRA dropout（必须与 SFT 一致）
LORA_TARGET="all"                                               # LoRA 目标模块（必须与 SFT 一致）

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    
    echo "=========================================="
    echo "开始 RL 训练"
    echo "=========================================="
    echo "SFT 模型: ${SFT_MODEL_PATH}"
    echo "训练文件: ${train_file}"
    echo "验证文件: ${eval_file}"
    echo "Info 文件: ${info_file}"
    echo "=========================================="
    
    python rl.py \
            --model_path ${SFT_MODEL_PATH} \
            --use_lora ${USE_LORA} \
            --lora_r ${LORA_R} \
            --lora_alpha ${LORA_ALPHA} \
            --lora_dropout ${LORA_DROPOUT} \
            --lora_target_modules ${LORA_TARGET} \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output/rl_lora_qwen25_3b \
            --info_file ${info_file} \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --category ${category} \
            --train_batch_size 32 \
            --eval_batch_size 32 \
            --num_generations 16 \
            --num_train_epochs 2 \
            --learning_rate 1e-5 \
            --beta 0.04 \
            --reward_type rule \
            --test_during_training True \
            --test_beam 20 \
            --wandb_project minionerec_lora \
            --wandb_run_name rl_lora_${category}
done

echo "=========================================="
echo "RL 训练完成！"
echo "模型保存在: ./output/rl_lora_qwen25_3b/final_checkpoint"
echo "=========================================="


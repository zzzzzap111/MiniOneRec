#!/bin/bash
# MiniOneRec SFT 训练脚本（支持 LoRA）

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export CUDA_VISIBLE_DEVICES=0   # 单卡训练，如果多卡请修改

# 配置参数
BASE_MODEL="./models/qwen3b"  
USE_LORA=True                           # 是否使用 LoRA
LORA_R=64                               # LoRA rank (增加到 64)
LORA_ALPHA=128                          # LoRA alpha (相应调整)
LORA_DROPOUT=0.05                       # LoRA dropout
LORA_TARGET="all"                       # LoRA 目标模块

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo "训练文件: ${train_file}"
    echo "验证文件: ${eval_file}"
    echo "索引文件: ./data/Amazon/index/${category}.index.json"
    echo "元数据文件: ./data/Amazon/index/${category}.item.json"
    
    # 单卡训练（推荐用于 LoRA）
    python sft.py \
            --base_model ${BASE_MODEL} \
            --use_lora ${USE_LORA} \
            --lora_r ${LORA_R} \
            --lora_alpha ${LORA_ALPHA} \
            --lora_dropout ${LORA_DROPOUT} \
            --lora_target_modules ${LORA_TARGET} \
            --batch_size 256 \
            --micro_batch_size 16 \
            --num_epochs 10 \
            --learning_rate 5e-4 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output/sft_lora_qwen25_3b \
            --wandb_project minionerec_lora \
            --wandb_run_name sft_lora_${category} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False
    
    # 如果需要多卡训练，使用以下命令（注释掉上面的 python 命令）
    # torchrun --nproc_per_node 8 \
    #         sft.py \
    #         --base_model ${BASE_MODEL} \
    #         --use_lora ${USE_LORA} \
    #         --lora_r ${LORA_R} \
    #         --lora_alpha ${LORA_ALPHA} \
    #         --lora_dropout ${LORA_DROPOUT} \
    #         --lora_target_modules ${LORA_TARGET} \
    #         --batch_size 1024 \
    #         --micro_batch_size 16 \
    #         --num_epochs 10 \
    #         --learning_rate 3e-4 \
    #         --train_file ${train_file} \
    #         --eval_file ${eval_file} \
    #         --output_dir ./output/sft_lora_qwen25_3b \
    #         --category ${category} \
    #         --train_from_scratch False \
    #         --seed 42 \
    #         --sid_index_path ./data/Amazon/index/${category}.index.json \
    #         --item_meta_path ./data/Amazon/index/${category}.item.json \
    #         --freeze_LLM False
done

echo "=========================================="
echo "SFT 训练完成！"
echo "模型保存在: ./output/sft_lora_qwen25_3b/final_checkpoint"
echo "=========================================="

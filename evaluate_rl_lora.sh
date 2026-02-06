#!/bin/bash

# ========================================
# LoRA 模型评估脚本 - RL 阶段
# ========================================

# 基础模型路径（原始模型）
BASE_MODEL="./models/qwen3b"

# LoRA 模型路径（RL 训练后的模型）
LORA_MODEL="./output/rl_lora_qwen25_3b/final_checkpoint"

# 数据集类别
CATEGORY="Industrial_and_Scientific"

# 数据路径
TEST_DATA="./data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"

# 评估参数
NUM_BEAMS=50
K=10
BATCH_SIZE=4

echo "========================================"
echo "评估 LoRA RL 模型"
echo "========================================"
echo "基础模型: ${BASE_MODEL}"
echo "LoRA 模型: ${LORA_MODEL}"
echo "测试数据: ${TEST_DATA}"
echo "信息文件: ${INFO_FILE}"
echo "类别: ${CATEGORY}"
echo "========================================"

python evaluate_lora.py \
    --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --test_data_path ${TEST_DATA} \
    --info_file ${INFO_FILE} \
    --category ${CATEGORY} \
    --num_beams ${NUM_BEAMS} \
    --K ${K} \
    --batch_size ${BATCH_SIZE}

echo "========================================"
echo "评估完成！"
echo "========================================"


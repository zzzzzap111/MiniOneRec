#!/bin/bash

# 检查训练状态和日志

echo "========================================"
echo "训练状态检查工具"
echo "========================================"
echo ""

# 1. 检查 SFT 输出目录
echo "1. 检查 SFT 输出目录"
echo "----------------------------------------"
if [ -d "./output/sft_lora_qwen25_3b" ]; then
    echo "✅ SFT 输出目录存在"
    echo ""
    echo "目录内容:"
    ls -lh ./output/sft_lora_qwen25_3b/
    echo ""
    
    # 检查 checkpoint
    if [ -d "./output/sft_lora_qwen25_3b/final_checkpoint" ]; then
        echo "✅ final_checkpoint 存在"
        echo ""
        echo "Checkpoint 内容:"
        ls -lh ./output/sft_lora_qwen25_3b/final_checkpoint/
        echo ""
    else
        echo "❌ final_checkpoint 不存在！"
    fi
    
    # 检查 adapter_config.json
    if [ -f "./output/sft_lora_qwen25_3b/final_checkpoint/adapter_config.json" ]; then
        echo "✅ LoRA adapter_config.json 存在"
        echo ""
        echo "LoRA 配置:"
        cat ./output/sft_lora_qwen25_3b/final_checkpoint/adapter_config.json
        echo ""
    else
        echo "❌ adapter_config.json 不存在！"
    fi
    
    # 检查 adapter_model.safetensors
    if [ -f "./output/sft_lora_qwen25_3b/final_checkpoint/adapter_model.safetensors" ]; then
        echo "✅ LoRA 权重文件存在"
        ls -lh ./output/sft_lora_qwen25_3b/final_checkpoint/adapter_model.safetensors
        echo ""
    else
        echo "❌ LoRA 权重文件不存在！"
    fi
else
    echo "❌ SFT 输出目录不存在！"
fi

echo ""
echo "========================================"
echo "2. 检查训练日志"
echo "----------------------------------------"

# 查找最近的日志文件
if [ -f "wandb/latest-run/logs/debug.log" ]; then
    echo "找到 wandb 日志"
    echo ""
    echo "最后 50 行训练日志:"
    tail -50 wandb/latest-run/logs/debug.log
elif [ -f "sft_training.log" ]; then
    echo "找到 sft_training.log"
    echo ""
    echo "最后 50 行训练日志:"
    tail -50 sft_training.log
else
    echo "⚠️  未找到训练日志文件"
    echo "建议: 重新训练时保存日志"
    echo "  bash sft.sh 2>&1 | tee sft_training.log"
fi

echo ""
echo "========================================"
echo "3. 检查训练损失"
echo "----------------------------------------"

# 尝试从 wandb 或日志中提取损失
if [ -d "./output/sft_lora_qwen25_3b" ]; then
    if [ -f "./output/sft_lora_qwen25_3b/trainer_state.json" ]; then
        echo "✅ 找到 trainer_state.json"
        echo ""
        echo "训练历史 (最后 10 个 epoch):"
        python -c "
import json
try:
    with open('./output/sft_lora_qwen25_3b/trainer_state.json', 'r') as f:
        state = json.load(f)
    
    if 'log_history' in state:
        logs = state['log_history']
        print(f'总共 {len(logs)} 条日志记录')
        print('')
        print('最后 10 条记录:')
        for log in logs[-10:]:
            if 'loss' in log:
                epoch = log.get('epoch', 'N/A')
                loss = log.get('loss', 'N/A')
                lr = log.get('learning_rate', 'N/A')
                print(f'  Epoch {epoch:.2f}: loss={loss:.4f}, lr={lr:.2e}')
            elif 'eval_loss' in log:
                epoch = log.get('epoch', 'N/A')
                eval_loss = log.get('eval_loss', 'N/A')
                print(f'  Epoch {epoch:.2f}: eval_loss={eval_loss:.4f}')
    else:
        print('❌ log_history 不存在')
except Exception as e:
    print(f'❌ 读取失败: {e}')
"
    else
        echo "❌ trainer_state.json 不存在"
    fi
else
    echo "❌ 无法检查训练状态"
fi

echo ""
echo "========================================"
echo "4. 检查数据加载"
echo "----------------------------------------"

# 检查数据文件
TRAIN_FILE="./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
INDEX_FILE="./data/Amazon/index/Industrial_and_Scientific.index.json"
ITEM_FILE="./data/Amazon/index/Industrial_and_Scientific.item.json"

if [ -f "$TRAIN_FILE" ]; then
    echo "✅ 训练数据存在"
    echo "   样本数: $(wc -l < $TRAIN_FILE)"
else
    echo "❌ 训练数据不存在: $TRAIN_FILE"
fi

if [ -f "$INDEX_FILE" ]; then
    echo "✅ SID 索引文件存在"
    echo "   SID 数量: $(wc -l < $INDEX_FILE)"
else
    echo "❌ SID 索引文件不存在: $INDEX_FILE"
fi

if [ -f "$ITEM_FILE" ]; then
    echo "✅ 商品元数据存在"
    echo "   商品数量: $(wc -l < $ITEM_FILE)"
else
    echo "❌ 商品元数据不存在: $ITEM_FILE"
fi

echo ""
echo "========================================"
echo "5. 诊断建议"
echo "----------------------------------------"

# 给出诊断建议
python -c "
import os
import json

issues = []
suggestions = []

# 检查 checkpoint
if not os.path.exists('./output/sft_lora_qwen25_3b/final_checkpoint'):
    issues.append('训练未完成或失败')
    suggestions.append('重新运行 SFT 训练: bash sft.sh 2>&1 | tee sft_training.log')

# 检查 LoRA 权重
if not os.path.exists('./output/sft_lora_qwen25_3b/final_checkpoint/adapter_model.safetensors'):
    issues.append('LoRA 权重文件缺失')
    suggestions.append('检查训练日志，确认训练是否正常完成')

# 检查训练状态
if os.path.exists('./output/sft_lora_qwen25_3b/trainer_state.json'):
    try:
        with open('./output/sft_lora_qwen25_3b/trainer_state.json', 'r') as f:
            state = json.load(f)
        
        if 'log_history' in state and len(state['log_history']) > 0:
            # 检查最终损失
            final_logs = [log for log in state['log_history'] if 'loss' in log]
            if final_logs:
                final_loss = final_logs[-1].get('loss', 999)
                if final_loss > 2.0:
                    issues.append(f'训练损失过高 ({final_loss:.4f})')
                    suggestions.append('训练可能未收敛，建议:')
                    suggestions.append('  - 增加训练轮数: --num_epochs 15')
                    suggestions.append('  - 降低学习率: --learning_rate 1e-4')
                    suggestions.append('  - 检查数据是否正确加载')
                elif final_loss < 0.1:
                    issues.append(f'训练损失异常低 ({final_loss:.4f})')
                    suggestions.append('可能过拟合或训练异常')
    except:
        pass

if issues:
    print('🔍 发现的问题:')
    for i, issue in enumerate(issues, 1):
        print(f'  {i}. {issue}')
    print('')
    print('💡 建议:')
    for suggestion in suggestions:
        print(f'  {suggestion}')
else:
    print('✅ 未发现明显问题')
    print('')
    print('💡 但评估效果很差 (HR@10 = 0.13%)，可能原因:')
    print('  1. 训练轮数不足 (当前默认 10 epochs)')
    print('  2. LoRA rank 太小 (当前 16)')
    print('  3. 学习率不合适')
    print('  4. 数据集太难或数据质量问题')
    print('')
    print('建议尝试:')
    print('  1. 增加训练轮数: --num_epochs 20')
    print('  2. 增加 LoRA rank: --lora_r 64')
    print('  3. 调整学习率: --learning_rate 5e-4')
    print('  4. 使用更大的模型 (如 Qwen2.5-7B)')
"

echo ""
echo "========================================"


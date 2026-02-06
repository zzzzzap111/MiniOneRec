#!/bin/bash

# 快速评估脚本 - 显示关键指标

RESULT_FILE=$1

if [ -z "$RESULT_FILE" ]; then
    echo "用法: bash quick_eval.sh <结果文件路径>"
    echo ""
    echo "示例:"
    echo "  bash quick_eval.sh ./output/sft_lora_qwen25_3b/eval_results.json"
    echo "  bash quick_eval.sh ./output/rl_lora_qwen25_3b/eval_results.json"
    exit 1
fi

if [ ! -f "$RESULT_FILE" ]; then
    echo "错误: 文件不存在: $RESULT_FILE"
    exit 1
fi

python -c "
import json
import math

with open('$RESULT_FILE', 'r') as f:
    results = json.load(f)

total = len(results)
hr1, hr5, hr10, hr20 = 0, 0, 0, 0
ndcg5, ndcg10, ndcg20 = 0, 0, 0
mrr = 0
ranks = []

for item in results:
    target = item['output'].strip()  # 去除换行符
    preds = [p.strip() for p in item['predict']]  # 去除换行符
    
    # HR@K
    if target in preds[:1]: hr1 += 1
    if target in preds[:5]: hr5 += 1
    if target in preds[:10]: hr10 += 1
    if target in preds[:20]: hr20 += 1
    
    # NDCG@K and MRR
    if target in preds[:20]:
        rank = preds[:20].index(target) + 1
        ranks.append(rank)
        
        if rank <= 5:
            ndcg5 += 1.0 / math.log2(rank + 1)
        if rank <= 10:
            ndcg10 += 1.0 / math.log2(rank + 1)
        ndcg20 += 1.0 / math.log2(rank + 1)
        
        mrr += 1.0 / rank

avg_rank = sum(ranks) / len(ranks) if ranks else 0

print('=' * 60)
print(f'评估结果: {total} 个测试样本')
print('=' * 60)
print()
print('📊 命中率 (Hit Rate):')
print(f'  HR@1:  {hr1/total:.4f} ({hr1/total*100:>5.2f}%)  ← Top-1 命中')
print(f'  HR@5:  {hr5/total:.4f} ({hr5/total*100:>5.2f}%)  ← Top-5 命中')
print(f'  HR@10: {hr10/total:.4f} ({hr10/total*100:>5.2f}%)  ← Top-10 命中')
print(f'  HR@20: {hr20/total:.4f} ({hr20/total*100:>5.2f}%)  ← Top-20 命中')
print()
print('📈 排名质量 (NDCG):')
print(f'  NDCG@5:  {ndcg5/total:.4f}')
print(f'  NDCG@10: {ndcg10/total:.4f}')
print(f'  NDCG@20: {ndcg20/total:.4f}')
print()
print('🎯 其他指标:')
print(f'  MRR (平均倒数排名): {mrr/total:.4f}')
print(f'  平均预测排名: {avg_rank:.2f}')
print()
print('=' * 60)

# 判断效果
hr10_val = hr10/total
ndcg10_val = ndcg10/total
mrr_val = mrr/total

if hr10_val >= 0.30 and ndcg10_val >= 0.18 and mrr_val >= 0.15:
    status = '✅ 优秀'
    color = '\033[92m'  # Green
elif hr10_val >= 0.25 and ndcg10_val >= 0.15 and mrr_val >= 0.12:
    status = '⚠️  良好'
    color = '\033[93m'  # Yellow
elif hr10_val >= 0.20 and ndcg10_val >= 0.12 and mrr_val >= 0.10:
    status = '⚠️  一般'
    color = '\033[93m'  # Yellow
else:
    status = '❌ 较差'
    color = '\033[91m'  # Red

reset = '\033[0m'

print(f'{color}训练效果评估: {status}{reset}')
print()

# 给出建议
if hr10_val < 0.25:
    print('💡 改进建议:')
    print('  - 增加训练轮数 (--num_epochs 15)')
    print('  - 增加 LoRA rank (--lora_r 32)')
    print('  - 调整学习率 (--learning_rate 5e-4)')
    print('  - 检查数据质量和预处理')
elif hr10_val >= 0.30:
    print('🎉 模型表现良好！可以考虑:')
    print('  - 继续 RL 训练以进一步提升')
    print('  - 在其他数据集上测试泛化能力')
    print('  - 部署到生产环境')

print('=' * 60)
"


"""
对比 SFT 和 RL 模型的评估结果
"""

import json
import math
import sys
import os

def load_results(file_path):
    """加载评估结果"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metrics(results, K=10):
    """计算评估指标"""
    total = len(results)
    hr_at_k = {1: 0, 5: 0, 10: 0, 20: 0}
    ndcg_at_k = {5: 0, 10: 0, 20: 0}
    mrr = 0
    ranks = []
    
    for item in results:
        target = item['output'].strip()  # 去除换行符
        predictions = [p.strip() for p in item['predict']]  # 去除换行符
        
        # HR@K
        for k in [1, 5, 10, 20]:
            if target in predictions[:k]:
                hr_at_k[k] += 1
        
        # NDCG@K and MRR
        if target in predictions[:20]:
            rank = predictions[:20].index(target) + 1
            ranks.append(rank)
            
            for k in [5, 10, 20]:
                if rank <= k:
                    ndcg_at_k[k] += 1.0 / math.log2(rank + 1)
            
            mrr += 1.0 / rank
    
    metrics = {}
    for k in [1, 5, 10, 20]:
        metrics[f'HR@{k}'] = hr_at_k[k] / total
    for k in [5, 10, 20]:
        metrics[f'NDCG@{k}'] = ndcg_at_k[k] / total
    metrics['MRR'] = mrr / total
    metrics['avg_rank'] = sum(ranks) / len(ranks) if ranks else 0
    
    return metrics

def print_comparison(sft_metrics, rl_metrics):
    """打印对比结果"""
    print("=" * 80)
    print("SFT vs RL 模型对比")
    print("=" * 80)
    print()
    
    # 命中率对比
    print("📊 命中率 (Hit Rate):")
    print(f"{'指标':<12} {'SFT 模型':<15} {'RL 模型':<15} {'绝对提升':<12} {'相对提升':<12}")
    print("-" * 80)
    for k in [1, 5, 10, 20]:
        key = f'HR@{k}'
        sft_val = sft_metrics[key]
        rl_val = rl_metrics[key]
        abs_improvement = rl_val - sft_val
        rel_improvement = ((rl_val - sft_val) / sft_val) * 100 if sft_val > 0 else 0
        
        # 颜色标记
        if rel_improvement > 8:
            symbol = '✅'
        elif rel_improvement > 3:
            symbol = '⚠️ '
        elif rel_improvement > 0:
            symbol = '  '
        else:
            symbol = '❌'
        
        print(f"{key:<12} {sft_val:<15.4f} {rl_val:<15.4f} {abs_improvement:>+11.4f} {symbol} {rel_improvement:>+6.2f}%")
    
    print()
    
    # NDCG 对比
    print("📈 排名质量 (NDCG):")
    print(f"{'指标':<12} {'SFT 模型':<15} {'RL 模型':<15} {'绝对提升':<12} {'相对提升':<12}")
    print("-" * 80)
    for k in [5, 10, 20]:
        key = f'NDCG@{k}'
        sft_val = sft_metrics[key]
        rl_val = rl_metrics[key]
        abs_improvement = rl_val - sft_val
        rel_improvement = ((rl_val - sft_val) / sft_val) * 100 if sft_val > 0 else 0
        
        if rel_improvement > 8:
            symbol = '✅'
        elif rel_improvement > 3:
            symbol = '⚠️ '
        elif rel_improvement > 0:
            symbol = '  '
        else:
            symbol = '❌'
        
        print(f"{key:<12} {sft_val:<15.4f} {rl_val:<15.4f} {abs_improvement:>+11.4f} {symbol} {rel_improvement:>+6.2f}%")
    
    print()
    
    # 其他指标
    print("🎯 其他指标:")
    print(f"{'指标':<12} {'SFT 模型':<15} {'RL 模型':<15} {'绝对提升':<12} {'相对提升':<12}")
    print("-" * 80)
    
    key = 'MRR'
    sft_val = sft_metrics[key]
    rl_val = rl_metrics[key]
    abs_improvement = rl_val - sft_val
    rel_improvement = ((rl_val - sft_val) / sft_val) * 100 if sft_val > 0 else 0
    
    if rel_improvement > 8:
        symbol = '✅'
    elif rel_improvement > 3:
        symbol = '⚠️ '
    elif rel_improvement > 0:
        symbol = '  '
    else:
        symbol = '❌'
    
    print(f"{key:<12} {sft_val:<15.4f} {rl_val:<15.4f} {abs_improvement:>+11.4f} {symbol} {rel_improvement:>+6.2f}%")
    
    key = 'avg_rank'
    sft_val = sft_metrics[key]
    rl_val = rl_metrics[key]
    abs_improvement = rl_val - sft_val
    rel_improvement = ((rl_val - sft_val) / sft_val) * 100 if sft_val > 0 else 0
    
    # 对于排名，越小越好
    if abs_improvement < -0.5:
        symbol = '✅'
    elif abs_improvement < 0:
        symbol = '⚠️ '
    elif abs_improvement == 0:
        symbol = '  '
    else:
        symbol = '❌'
    
    print(f"{'平均排名':<12} {sft_val:<15.2f} {rl_val:<15.2f} {abs_improvement:>+11.2f} {symbol} {rel_improvement:>+6.2f}%")
    
    print()
    print("=" * 80)
    
    # 总体评估
    hr10_improvement = ((rl_metrics['HR@10'] - sft_metrics['HR@10']) / sft_metrics['HR@10']) * 100
    
    print()
    print("🏆 总体评估:")
    if hr10_improvement > 8:
        print("  ✅ RL 训练效果显著！HR@10 提升超过 8%")
        print("  建议: 可以部署 RL 模型到生产环境")
    elif hr10_improvement > 3:
        print("  ⚠️  RL 训练有一定效果，HR@10 提升 3-8%")
        print("  建议: 可以尝试调整 RL 超参数以获得更好效果")
    elif hr10_improvement > 0:
        print("  ⚠️  RL 训练效果不明显，HR@10 提升小于 3%")
        print("  建议: 检查 RL 配置，尝试调整 beta、learning_rate 等参数")
    else:
        print("  ❌ RL 训练失败，性能反而下降")
        print("  建议: 检查 RL 配置，可能需要:")
        print("    - 降低学习率 (--learning_rate 5e-7)")
        print("    - 调整 beta 参数 (--beta 0.02 或 0.08)")
        print("    - 增加训练轮数 (--num_train_epochs 2)")
    
    print("=" * 80)

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python compare_results.py <SFT结果文件> <RL结果文件>")
        print()
        print("示例:")
        print("  python compare_results.py \\")
        print("    ./output/sft_lora_qwen25_3b/eval_results.json \\")
        print("    ./output/rl_lora_qwen25_3b/eval_results.json")
        sys.exit(1)
    
    sft_file = sys.argv[1]
    rl_file = sys.argv[2]
    
    print(f"\n加载 SFT 结果: {sft_file}")
    sft_results = load_results(sft_file)
    print(f"SFT 样本数: {len(sft_results)}")
    
    print(f"\n加载 RL 结果: {rl_file}")
    rl_results = load_results(rl_file)
    print(f"RL 样本数: {len(rl_results)}")
    
    if len(sft_results) != len(rl_results):
        print("\n警告: SFT 和 RL 的样本数不一致！")
    
    print("\n计算指标...")
    sft_metrics = calculate_metrics(sft_results, K=10)
    rl_metrics = calculate_metrics(rl_results, K=10)
    
    print()
    print_comparison(sft_metrics, rl_metrics)

if __name__ == "__main__":
    main()


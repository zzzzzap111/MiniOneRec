"""
诊断评估结果文件，找出问题所在
"""

import json
import sys

def diagnose_results(file_path):
    """诊断评估结果"""
    print("=" * 60)
    print("评估结果诊断工具")
    print("=" * 60)
    print()
    
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ 错误: 无法读取文件: {e}")
        return
    
    print(f"✅ 文件读取成功")
    print(f"📊 样本总数: {len(results)}")
    print()
    
    # 检查前几个样本
    print("=" * 60)
    print("检查前 3 个样本:")
    print("=" * 60)
    
    for i, item in enumerate(results[:3]):
        print(f"\n样本 {i+1}:")
        print("-" * 60)
        
        # 检查字段
        if 'input' in item:
            print(f"✅ input 字段存在")
            print(f"   内容预览: {item['input'][:100]}...")
        else:
            print(f"❌ input 字段缺失")
        
        if 'output' in item:
            print(f"✅ output 字段存在")
            print(f"   真实目标: {item['output']}")
        else:
            print(f"❌ output 字段缺失")
        
        if 'predict' in item:
            print(f"✅ predict 字段存在")
            print(f"   预测类型: {type(item['predict'])}")
            
            if isinstance(item['predict'], list):
                print(f"   预测数量: {len(item['predict'])}")
                if len(item['predict']) > 0:
                    print(f"   Top-1 预测: {item['predict'][0]}")
                    print(f"   Top-5 预测: {item['predict'][:5]}")
                else:
                    print(f"   ❌ 预测列表为空！")
            else:
                print(f"   ❌ 预测不是列表类型！")
                print(f"   预测内容: {item['predict']}")
        else:
            print(f"❌ predict 字段缺失")
    
    print()
    print("=" * 60)
    print("统计分析:")
    print("=" * 60)
    
    # 统计问题
    missing_output = 0
    missing_predict = 0
    empty_predict = 0
    wrong_type_predict = 0
    has_match = 0
    
    for item in results:
        if 'output' not in item:
            missing_output += 1
            continue
        
        if 'predict' not in item:
            missing_predict += 1
            continue
        
        if not isinstance(item['predict'], list):
            wrong_type_predict += 1
            continue
        
        if len(item['predict']) == 0:
            empty_predict += 1
            continue
        
        # 检查是否有匹配
        target = item['output']
        predictions = item['predict']
        if target in predictions:
            has_match += 1
    
    print(f"缺失 output 字段: {missing_output} ({missing_output/len(results)*100:.2f}%)")
    print(f"缺失 predict 字段: {missing_predict} ({missing_predict/len(results)*100:.2f}%)")
    print(f"predict 类型错误: {wrong_type_predict} ({wrong_type_predict/len(results)*100:.2f}%)")
    print(f"predict 列表为空: {empty_predict} ({empty_predict/len(results)*100:.2f}%)")
    print(f"有匹配的样本: {has_match} ({has_match/len(results)*100:.2f}%)")
    
    print()
    print("=" * 60)
    print("问题诊断:")
    print("=" * 60)
    
    if missing_output > 0:
        print(f"❌ 问题 1: {missing_output} 个样本缺失 output 字段")
        print(f"   原因: 数据集格式问题")
        print(f"   解决: 检查测试数据文件")
    
    if missing_predict > 0:
        print(f"❌ 问题 2: {missing_predict} 个样本缺失 predict 字段")
        print(f"   原因: 评估脚本未正确生成预测")
        print(f"   解决: 检查评估脚本是否正常运行")
    
    if wrong_type_predict > 0:
        print(f"❌ 问题 3: {wrong_type_predict} 个样本的 predict 不是列表")
        print(f"   原因: 评估脚本输出格式错误")
        print(f"   解决: 检查评估脚本的输出格式")
    
    if empty_predict > 0:
        print(f"❌ 问题 4: {empty_predict} 个样本的 predict 列表为空")
        print(f"   原因: 模型未生成任何预测")
        print(f"   解决: 检查模型加载和生成配置")
    
    if has_match == 0 and empty_predict == 0:
        print(f"❌ 问题 5: 所有预测都不匹配目标")
        print(f"   原因: 可能的原因:")
        print(f"   1. 模型训练失败，未学到有效模式")
        print(f"   2. 预测格式与目标格式不一致")
        print(f"   3. tokenizer 或 SID 映射问题")
        print(f"   解决:")
        print(f"   1. 检查训练日志，确认训练是否正常")
        print(f"   2. 比较 predict 和 output 的格式")
        print(f"   3. 检查 SID index 文件是否正确")
    
    if has_match > 0:
        print(f"✅ 有 {has_match} 个样本匹配成功")
        print(f"   HR@{len(results[0]['predict']) if 'predict' in results[0] and isinstance(results[0]['predict'], list) else 'N/A'}: {has_match/len(results):.4f}")
    
    print()
    print("=" * 60)
    print("详细检查: 比较预测和目标格式")
    print("=" * 60)
    
    # 检查格式差异
    for i, item in enumerate(results[:5]):
        if 'output' in item and 'predict' in item and isinstance(item['predict'], list) and len(item['predict']) > 0:
            print(f"\n样本 {i+1}:")
            print(f"  目标格式: '{item['output']}'")
            print(f"  目标长度: {len(item['output'])}")
            print(f"  预测格式: '{item['predict'][0]}'")
            print(f"  预测长度: {len(item['predict'][0])}")
            
            # 检查是否只是空格或格式问题
            if item['output'].strip() == item['predict'][0].strip():
                print(f"  ⚠️  去除空格后匹配！可能是空格问题")
            elif item['output'].lower() == item['predict'][0].lower():
                print(f"  ⚠️  忽略大小写后匹配！可能是大小写问题")
            else:
                print(f"  ❌ 完全不匹配")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_results.py <评估结果文件>")
        print()
        print("示例:")
        print("  python diagnose_results.py ./output/sft_lora_qwen25_3b/eval_results.json")
        sys.exit(1)
    
    file_path = sys.argv[1]
    diagnose_results(file_path)


"""
LoRA 模型评估脚本
用于评估使用 LoRA 微调的 MiniOneRec 模型
"""

import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from peft import PeftModel
from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
import random
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(
    base_model: str = "",  # 原始基础模型路径，如 ./models/qwen3b
    lora_model: str = "",  # LoRA 模型路径，如 ./output/sft_lora_qwen25_3b/final_checkpoint
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 10,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
):
    """
    评估 LoRA 微调后的模型
    
    Args:
        base_model: 原始基础模型路径（如 ./models/qwen3b 或 Qwen/Qwen2.5-3B-Instruct）
        lora_model: LoRA 模型路径（如 ./output/sft_lora_qwen25_3b/final_checkpoint）
        info_file: 商品信息文件
        category: 商品类别
        test_data_path: 测试数据路径
        result_json_data: 结果保存路径
        batch_size: 批次大小
        K: Top-K 评估
        seed: 随机种子
        length_penalty: 长度惩罚
        max_new_tokens: 最大生成 token 数
        num_beams: 束搜索大小
    """
    random.seed(seed)
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books"
    }
    category_name = category_dict[category]
    print(f"类别: {category_name}")
    
    # 1. 加载基础模型
    print(f"\n加载基础模型: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. 加载 LoRA tokenizer（包含新 tokens）
    print(f"加载 LoRA tokenizer: {lora_model}")
    tokenizer = AutoTokenizer.from_pretrained(lora_model)
    
    # 3. 调整基础模型的词表大小
    print(f"原始词表大小: {model.get_input_embeddings().weight.shape[0]}")
    print(f"新词表大小: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"调整后词表大小: {model.get_input_embeddings().weight.shape[0]}")
    
    # 4. 加载 LoRA 适配器
    print(f"\n加载 LoRA 适配器: {lora_model}")
    model = PeftModel.from_pretrained(model, lora_model)
    model.eval()
    print("模型加载完成！\n")
    
    # 5. 加载商品信息
    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]
    
    # 6. 创建 prefixID
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    # 7. 构建哈希字典
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])
    
    # Convert sets to lists
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    
    # 8. 定义前缀约束函数
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
    
    # 9. 设置 tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # 10. 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    val_dataset = EvalSidDataset(
        train_file=test_data_path,
        tokenizer=tokenizer,
        max_len=2560,
        category=category_name,
        test=True,
        K=K,
        seed=seed
    )
    print(f"测试样本数: {len(val_dataset)}\n")
    
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()
    
    # 11. 设置模型配置
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    # 12. 定义评估函数
    def evaluate(
        encodings,
        num_beams=10,
        max_new_tokens=64,
        length_penalty=1.0,
        **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])
        
        padding_encodings = {"input_ids": []}
        attention_mask = []
        
        for _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L)
        
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model
            )
            logits_processor = LogitsProcessorList([clp])
            
            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
        
        batched_completions = generation_output.sequences[:, maxLen:]
        
        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
        
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    # 13. 执行评估
    model = model.to(device)
    
    print("开始评估...")
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])
    
    for idx, encodings_batch in enumerate(tqdm(new_encodings)):
        output = evaluate(encodings_batch, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty)
        outputs = outputs + output
    
    # 14. 保存结果
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
    
    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')
    
    # 如果没有指定结果文件，使用默认路径
    if not result_json_data:
        result_json_data = lora_model.replace("/final_checkpoint", "") + "/eval_results.json"
    
    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"\n结果已保存到: {result_json_data}")
    print("\n评估完成！")

if __name__ == "__main__":
    fire.Fire(main)

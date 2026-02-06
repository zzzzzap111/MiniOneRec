"""
LoRA/QLoRA 代码修改示例

本文件展示了如何在 sft.py 和 rl.py 中添加 LoRA/QLoRA 支持。
请参考这些示例修改对应的文件。
"""

# ============================================================================
# 1. sft.py 修改示例
# ============================================================================

# 步骤 1：在文件开头添加导入
"""
在 sft.py 文件开头（import 部分）添加：
"""
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# 步骤 2：在 train() 函数签名中添加参数
"""
修改 train() 函数签名，添加以下参数：
"""
def train(
    # ... 现有参数保持不变 ...
    use_lora: bool = False,  # 是否使用 LoRA
    use_qlora: bool = False,  # 是否使用 QLoRA
    lora_r: int = 16,  # LoRA rank
    lora_alpha: int = 32,  # LoRA alpha
    lora_dropout: float = 0.05,  # LoRA dropout
    lora_target_modules: str = "all",  # 目标模块："all", "qkv", 或自定义列表
    # ... 其他现有参数 ...
):
    # ... 现有代码保持不变，直到模型加载部分 ...
    
    # 步骤 3：修改模型加载部分（在 if not train_from_scratch: 之后）
    """
    将原来的：
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    
    替换为：
    """
    if not train_from_scratch:
        if use_qlora:
            print("=" * 50)
            print("启用 QLoRA 微调（4-bit 量化 + LoRA）")
            print("=" * 50)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
            )
    else:
        # train_from_scratch 的情况保持不变
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
    
    # ... tokenizer 设置和 SID tokens 添加保持不变 ...
    
    # 步骤 4：在添加 SID tokens 之后，添加 LoRA 配置
    """
    在 model.resize_token_embeddings(len(tokenizer)) 之后添加：
    """
    if sid_index_path and os.path.exists(sid_index_path):
        # ... 现有代码 ...
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
    
    # ========== 添加 LoRA/QLoRA 配置 ==========
    if use_lora or use_qlora:
        print("=" * 50)
        if use_qlora:
            print("配置 QLoRA（LoRA + 4-bit 量化）")
        else:
            print("配置 LoRA")
        print("=" * 50)
        
        # 确定目标模块
        if lora_target_modules == "all":
            # 自动检测模型结构
            if hasattr(model, 'model'):
                model_base = model.model
            else:
                model_base = model
            
            target_modules = []
            for name, module in model_base.named_modules():
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", 
                                            "gate_proj", "up_proj", "down_proj"]):
                    target_modules.append(name.split('.')[-1])
            
            target_modules = list(set(target_modules))
            if not target_modules:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif lora_target_modules == "qkv":
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            target_modules = [m.strip() for m in lora_target_modules.split(",")]
        
        print(f"LoRA 目标模块: {target_modules}")
        
        # 创建 LoRA 配置
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用 LoRA
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.4f}%)")
        print("=" * 50)
    
    # ... 后续代码（Freeze LLM 参数等）保持不变 ...


# ============================================================================
# 2. rl.py 修改示例
# ============================================================================

# 步骤 1：在文件开头添加导入
"""
在 rl.py 文件开头添加：
"""
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType

# 步骤 2：在 train() 函数签名中添加参数
"""
修改 train() 函数签名，添加以下参数：
"""
def train(
    # ... 现有参数保持不变 ...
    use_lora: bool = False,  # 是否使用 LoRA
    use_qlora: bool = False,  # 是否使用 QLoRA
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all",
    # ... 其他现有参数 ...
):
    # ... 现有代码保持不变，直到创建 ReReTrainer 之前 ...
    
    # 步骤 3：在创建 ReReTrainer 之前，添加 LoRA 配置
    """
    在 trainer = ReReTrainer(...) 之前添加：
    """
    # ========== 配置 LoRA/QLoRA ==========
    peft_config = None
    if use_lora or use_qlora:
        print("=" * 50)
        if use_qlora:
            print("RL 阶段启用 QLoRA 微调（4-bit 量化 + LoRA）")
        else:
            print("RL 阶段启用 LoRA 微调")
        print("=" * 50)
        
        # 确定目标模块（需要先加载模型来检测结构）
        if lora_target_modules == "all":
            from transformers import AutoModelForCausalLM
            temp_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            if hasattr(temp_model, 'model'):
                model_base = temp_model.model
            else:
                model_base = temp_model
            
            target_modules = []
            for name, module in model_base.named_modules():
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"]):
                    target_modules.append(name.split('.')[-1])
            target_modules = list(set(target_modules))
            if not target_modules:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            del temp_model
            torch.cuda.empty_cache()  # 清理显存
        elif lora_target_modules == "qkv":
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            target_modules = [m.strip() for m in lora_target_modules.split(",")]
        
        print(f"LoRA 目标模块: {target_modules}")
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        print("=" * 50)
    
    # 步骤 4：修改 ReReTrainer 初始化，传入 peft_config
    """
    修改 trainer = ReReTrainer(...) 调用，添加 peft_config 参数：
    """
    # 如果使用 QLoRA，需要在 GRPOConfig 中设置量化配置
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # 注意：需要在 training_args 创建时设置 model_init_kwargs
        # 或者在 ReReTrainer 内部处理
    
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        peft_config=peft_config,  # 传入 LoRA 配置
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )


# ============================================================================
# 3. 使用示例
# ============================================================================

"""
使用 LoRA 训练 SFT：
python sft.py \
    --base_model Qwen/Qwen2-3B-Instruct \
    --train_file train.csv \
    --eval_file valid.csv \
    --output_dir ./output/sft_lora \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --batch_size 512 \
    --micro_batch_size 8 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/index.json \
    --item_meta_path ./data/item.json

使用 QLoRA 训练 SFT：
python sft.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --train_file train.csv \
    --eval_file valid.csv \
    --output_dir ./output/sft_qlora \
    --use_qlora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --batch_size 512 \
    --micro_batch_size 16 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/index.json \
    --item_meta_path ./data/item.json

使用 LoRA 训练 RL：
python rl.py \
    --model_path ./output/sft_lora/final_checkpoint \
    --train_file train.csv \
    --eval_file valid.csv \
    --output_dir ./output/rl_lora \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --category Industrial_and_Scientific \
    --info_file ./data/info.json \
    --sid_index_path ./data/index.json \
    --item_meta_path ./data/item.json
"""


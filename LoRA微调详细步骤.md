# MiniOneRec LoRA 微调详细步骤（A100 云服务器）

## 🚀 快速开始（完整流程）

### 当前状态确认

您的情况：
- ✅ 云服务器：A100 GPU
- ✅ 模型：Qwen2.5-3B-Instruct（已下载）
- ✅ 环境：已配置（`pip install -r requirements.txt`）
- ✅ 微调方式：LoRA
- ✅ 代码：已修改支持 LoRA

### 完整训练流程（4步）

**步骤 0：修改代码支持 LoRA（5分钟，只需一次）**

⚠️ **重要**：原始代码不支持 LoRA，需要先修改 `sft.py` 和 `rl.py`。

**快速修改方法**：

1. 在 `sft.py` 文件开头（第 23 行附近）添加导入：
```python
from peft import LoraConfig, get_peft_model, TaskType
```

2. 在 `sft.py` 的 `train()` 函数参数中（第 108 行附近）添加 LoRA 参数：
```python
def train(
    # ... 现有参数 ...
    freeze_LLM: bool = False,
    # LoRA params（新增以下5行）
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all",
    # wandb params
    wandb_project: str = "",
    # ... 其他参数 ...
):
```

3. 在 `sft.py` 的模型加载后（第 166 行附近，`model.resize_token_embeddings(len(tokenizer))` 之后）添加 LoRA 配置：
```python
    # ========== 添加 LoRA 配置 ==========
    if use_lora:
        print("=" * 50)
        print("启用 LoRA 微调")
        print("=" * 50)
        
        # 确定目标模块
        if lora_target_modules == "all":
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
```

4. 同样修改 `rl.py`（参考下面的"详细代码修改"部分）

**或者使用我提供的修改示例**：
- 参考 `LoRA_代码修改示例.py` 文件
- 或查看下面的"步骤 3：修改代码以支持 LoRA"部分

**步骤 1：确认数据和模型位置（5分钟）**

```bash
# 1. 进入项目目录
cd /path/to/MiniOneRec

# 2. 检查数据文件
ls -lh ./data/Amazon/train/Industrial_and_Scientific*
ls -lh ./data/Amazon/valid/Industrial_and_Scientific*
ls -lh ./data/Amazon/index/Industrial_and_Scientific*

# 3. 确认模型位置
# 如果模型在默认位置：~/.cache/huggingface/hub/
# 直接使用：Qwen/Qwen2.5-3B-Instruct
# 如果模型在自定义目录：./models/Qwen2.5-3B-Instruct
# 使用完整路径
```

**步骤 2：运行 SFT 训练（2-4小时）**

```bash
python sft.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/sft_lora_qwen25_3b \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --batch_size 1024 \
    --micro_batch_size 32 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --wandb_project minionerec_lora \
    --wandb_run_name sft_lora_qwen25_3b
```

**步骤 3：运行 RL 训练（1-2小时）**

```bash
python rl.py \
    --model_path ./output/sft_lora_qwen25_3b/final_checkpoint \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/rl_lora_qwen25_3b \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --category Industrial_and_Scientific \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_generations 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --reward_type rule \
    --wandb_project minionerec_lora \
    --wandb_run_name rl_lora_qwen25_3b
```

**步骤 4：评估模型性能（10分钟）**

```bash
python evaluate.py \
    --exp_name ./output/rl_lora_qwen25_3b/final_checkpoint \
    --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --num_beams 50 \
    --K 10
```

### 监控训练

在另一个终端监控GPU使用：

```bash
watch -n 1 nvidia-smi
```

### 预期输出

**SFT训练开始时**：
```
==================================================
启用 LoRA 微调
==================================================
LoRA 目标模块: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
可训练参数: 13,107,200 / 3,000,000,000 (0.44%)
==================================================
Loading index from ./data/Amazon/index/Industrial_and_Scientific.index.json
Adding 765 new tokens to tokenizer
```

**训练时间估算**：
- SFT训练：约 2-4 小时（Qwen2.5-3B + LoRA + A100）
- RL训练：约 1-2 小时
- 总计：约 3-6 小时

---

## 📋 问题回答

### 1. 当前项目目录下有基础模型吗？

**答案：没有**

- 项目目录下**没有基础模型文件**（如 `.pth`、`.bin` 等）
- 基础模型需要从 **HuggingFace** 下载
- 模型会在首次使用时自动下载，或可以提前下载

### 2. LoRA 微调详细步骤

由于你使用的是 **A100 GPU**，推荐使用较大的模型（如 Qwen2-7B），LoRA 可以显著节省显存。

---

## ⚠️ 重要提示：代码需要修改

**原始代码不支持 LoRA**，需要先修改 `sft.py` 和 `rl.py` 才能使用 `--use_lora True` 参数。

有两种方式：
1. **手动修改**：参考下面"步骤 3：修改代码以支持 LoRA"部分
2. **查看示例**：参考 `LoRA_代码修改示例.py` 文件

修改完成后，才能运行"快速开始"部分的命令。

---

## 🚀 完整步骤（A100 环境）

### 步骤 1：确认环境

```bash
# 检查 GPU
nvidia-smi

# 检查 Python 环境
python --version  # 应该是 3.10+

# 检查依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# 检查 huggingface-hub（用于自动下载模型）
python -c "from huggingface_hub import __version__; print(f'huggingface-hub: {__version__}')"
```

**重要提示**：
- ✅ **不需要手动下载模型**，代码会在首次运行时自动从 HuggingFace 下载
- ✅ 模型会下载到 `~/.cache/huggingface/hub/` 目录
- ✅ 如果 HuggingFace 访问慢，可以设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 步骤 2：下载基础模型（不需要手动下载！）

**✅ 重要：基础模型不需要手动下载，代码会自动下载！**

#### 推荐方式：代码自动下载（最简单）

**直接运行训练命令即可**，代码会在首次运行时自动从 HuggingFace 下载模型：

```bash
# 直接运行，模型会自动下载到 ~/.cache/huggingface/hub/
python sft.py --base_model Qwen/Qwen2-7B-Instruct ...
```

**模型下载位置**：
- 自动保存到：`~/.cache/huggingface/hub/`
- 下载一次后，后续运行会直接使用缓存
- 不需要手动管理模型文件

#### 如果 HuggingFace 访问慢（设置镜像）

```bash
# 设置国内镜像（在运行训练前）
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常运行训练命令
python sft.py --base_model Qwen/Qwen2-7B-Instruct ...
```

#### 如果需要手动下载（可选）

**方式 1：使用 Python 代码下载**

创建 `download_model.py`：

```python
from huggingface_hub import snapshot_download

# 下载模型到指定目录
snapshot_download(
    repo_id="Qwen/Qwen2-7B-Instruct",
    local_dir="./models/Qwen2-7B-Instruct",
    local_dir_use_symlinks=False
)
print("模型下载完成！")
```

运行：
```bash
python download_model.py
```

**方式 2：使用 Python 模块方式调用 huggingface-cli**

如果 `huggingface-cli` 命令找不到，可以使用 Python 模块方式：

```bash
# 使用 Python 模块方式调用
python -m huggingface_hub.cli.download \
    Qwen/Qwen2-7B-Instruct \
    --local-dir ./models/Qwen2-7B-Instruct
```

**注意**：即使 `huggingface-cli` 命令不可用，代码也会自动下载模型，**不需要手动下载**！

**推荐模型（A100 推荐）**：

| 模型 | 推荐度 | 显存占用 (LoRA) | 适用场景 |
|------|--------|----------------|---------|
| `Qwen/Qwen2.5-7B-Instruct` | ⭐⭐⭐⭐⭐ | ~12GB | **A100 最佳选择**（性能与资源平衡） |
| `Qwen/Qwen2.5-3B-Instruct` | ⭐⭐⭐⭐ | ~8GB | 显存受限或需要更快训练 |
| `Qwen/Qwen2-7B-Instruct` | ⭐⭐⭐⭐ | ~12GB | 稳定可靠的选择 |
| `Qwen/Qwen2-3B-Instruct` | ⭐⭐⭐ | ~8GB | 资源受限场景 |
| `Qwen/Qwen2.5-14B-Instruct` | ⭐⭐⭐⭐⭐ | ~24GB | 多卡 A100，追求最佳性能 |

**选择建议**：
- ✅ **A100 单卡推荐**：`Qwen2.5-7B-Instruct`（性能最好，显存充足）
- ✅ **如果选择 3B**：`Qwen2.5-3B-Instruct`（比 Qwen2-3B 性能更好）
- ✅ **Qwen2.5 系列**是 Qwen2 的升级版，**强烈推荐使用新版本**

### 步骤 3：修改代码以支持 LoRA

#### 3.1 修改 `sft.py`

在 `sft.py` 中添加 LoRA 支持：

```python
# 在文件开头（第 23 行附近）添加导入
from peft import LoraConfig, get_peft_model, TaskType
```

在 `train()` 函数中添加参数（第 90 行附近）：

```python
def train(
    # model/data params
    base_model: str = "",
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,
    freeze_LLM: bool = False,
    # LoRA 参数（新增）
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all",
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
```

在模型加载和 SID tokens 添加之后（第 159 行之后），添加 LoRA 配置：

```python
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # ========== 添加 LoRA 配置 ==========
    if use_lora:
        print("=" * 50)
        print("启用 LoRA 微调")
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

    # Freeze LLM parameters if required
    if freeze_LLM:
        # ... 现有代码保持不变 ...
```

#### 3.2 修改 `rl.py`

在 `rl.py` 中添加 LoRA 支持：

```python
# 在文件开头（第 8 行附近）添加导入
from peft import LoraConfig, TaskType
```

在 `train()` 函数中添加参数（第 30 行附近）：

```python
def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    # LoRA 参数（新增）
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "all",
):
```

在创建 `ReReTrainer` 之前（第 288 行之前），添加 LoRA 配置：

```python
    # ========== 配置 LoRA ==========
    peft_config = None
    if use_lora:
        print("=" * 50)
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

    training_args = GRPOConfig(output_dir=output_dir,
                                # ... 现有参数 ...
                            )
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
```

### 步骤 4：准备数据

确认数据文件存在：

```bash
# 检查数据文件
ls -lh ./data/Amazon/index/Industrial_and_Scientific.*
ls -lh ./data/Amazon/train/Industrial_and_Scientific*
ls -lh ./data/Amazon/valid/Industrial_and_Scientific*
ls -lh ./data/Amazon/info/Industrial_and_Scientific*
```

应该看到：
- `Industrial_and_Scientific.item.json`
- `Industrial_and_Scientific.index.json`
- `Industrial_and_Scientific_5_2016-10-2018-11.csv` (train/valid/test)
- `Industrial_and_Scientific_5_2016-10-2018-11.txt` (info)

### 步骤 5：运行 LoRA 微调

#### 5.1 SFT 阶段（使用 LoRA）

**单卡训练**（A100 80GB 推荐配置）：

```bash
python sft.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/sft_lora \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --batch_size 1024 \
    --micro_batch_size 16 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --wandb_project minionerec_lora \
    --wandb_run_name sft_lora_qwen7b
```

**多卡训练**（如果有多个 A100）：

```bash
torchrun --nproc_per_node 8 \
    sft.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/sft_lora \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --batch_size 1024 \
    --micro_batch_size 16 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --wandb_project minionerec_lora \
    --wandb_run_name sft_lora_qwen7b
```

**预期输出**：
```
==================================================
启用 LoRA 微调
==================================================
LoRA 目标模块: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
可训练参数: 13,107,200 / 7,000,000,000 (0.19%)
==================================================
```

#### 5.2 RL 阶段（使用 LoRA）

**重要**：RL 阶段必须使用 SFT 阶段训练好的模型，且 LoRA 配置需要一致。

```bash
python rl.py \
    --model_path ./output/sft_lora/final_checkpoint \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/rl_lora \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --category Industrial_and_Scientific \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_generations 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --reward_type rule \
    --wandb_project minionerec_lora \
    --wandb_run_name rl_lora_qwen7b
```

### 步骤 6：验证训练结果

#### 6.1 检查输出文件

```bash
# SFT 输出
ls -lh ./output/sft_lora/
# 应该看到：
# - adapter_config.json（LoRA 配置）
# - adapter_model.bin（LoRA 权重）
# - final_checkpoint/（完整检查点）

# RL 输出
ls -lh ./output/rl_lora/
```

#### 6.2 加载 LoRA 模型进行推理

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(
    base_model,
    "./output/sft_lora/final_checkpoint"
)

# 合并 LoRA 权重（可选，用于推理加速）
# model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("./output/sft_lora/final_checkpoint")
```

---

## 📊 A100 推荐配置

### 单卡 A100 (80GB)

| 模型 | LoRA 显存占用 | 推荐 batch_size | 推荐 micro_batch_size | 推荐度 |
|------|-------------|----------------|---------------------|--------|
| **Qwen2.5-7B** | ~12GB | 1024 | 16 | ⭐⭐⭐⭐⭐ **最佳** |
| **Qwen2.5-3B** | ~8GB | 1024 | 16-32 | ⭐⭐⭐⭐ **快速训练** |
| Qwen2-7B | ~12GB | 1024 | 16 | ⭐⭐⭐⭐ |
| Qwen2-3B | ~8GB | 1024 | 16-32 | ⭐⭐⭐ |
| Qwen2-14B | ~24GB | 512 | 8-16 | ⭐⭐⭐ |

### 多卡 A100

使用 `torchrun` 进行多卡训练：

```bash
# 8 卡训练
torchrun --nproc_per_node 8 sft.py ...
```

---

## ⚙️ LoRA 参数调优建议

### 推荐配置（A100）

| 场景 | r | alpha | dropout | target_modules |
|------|---|-------|---------|----------------|
| **快速实验** | 8 | 16 | 0.05 | "qkv" |
| **平衡性能** | 16 | 32 | 0.05 | "all" ✅ **推荐** |
| **最佳性能** | 32 | 64 | 0.1 | "all" |

### 参数说明

- **`r`**：LoRA rank，控制适配器大小
  - 较小值（8-16）：显存占用更少，但可能欠拟合
  - 较大值（32-64）：性能更好，但显存占用增加
  - **推荐**：16（平衡性能和资源）

- **`lora_alpha`**：LoRA 的缩放因子
  - 通常设置为 `r` 的 2 倍
  - **推荐**：`alpha = 2 * r`（即 32）

- **`lora_dropout`**：Dropout 率
  - **推荐**：0.05-0.1

- **`lora_target_modules`**：目标模块
  - `"all"`：自动检测所有注意力层和 MLP 层（推荐）
  - `"qkv"`：只针对注意力层（q_proj, k_proj, v_proj, o_proj）
  - **推荐**：`"all"`（性能最好）

---

## 🔍 常见问题

### 1. huggingface-cli 命令未找到

**问题**：`huggingface-cli: command not found`

**解决方案**：

**✅ 最简单的方法：不需要使用 huggingface-cli！**

代码会自动下载模型，直接运行训练命令即可：

```bash
python sft.py --base_model Qwen/Qwen2-7B-Instruct ...
```

**如果需要手动下载，使用 Python 模块方式**：

```bash
# 方式 1：使用 Python 模块调用（推荐）
python -m huggingface_hub.cli.download \
    Qwen/Qwen2-7B-Instruct \
    --local-dir ./models/Qwen2-7B-Instruct

# 方式 2：使用 Python 代码
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B-Instruct', local_dir='./models/Qwen2-7B-Instruct')"
```

**如果 HuggingFace 访问慢**：

```bash
# 设置镜像环境变量（推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常运行训练命令
python sft.py --base_model Qwen/Qwen2-7B-Instruct ...
```

**重要**：即使 `huggingface-cli` 命令不可用，代码也会自动下载模型，**不需要手动下载**！

### 2. 显存不足（OOM）

**解决方案**：
- 减小 `batch_size` 或 `micro_batch_size`
- 使用更小的 `lora_r`（如 8）
- 使用 `lora_target_modules="qkv"`（只训练注意力层）

### 3. LoRA 训练后如何加载

```python
# 方式 1：加载 LoRA 适配器（推荐）
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./output/sft_lora/final_checkpoint")

# 方式 2：合并权重（用于推理加速）
model = model.merge_and_unload()
model.save_pretrained("./output/sft_lora_merged")
```

### 4. SFT 和 RL 阶段 LoRA 配置不一致

**重要**：SFT 和 RL 阶段的 LoRA 配置（`r`, `alpha`, `target_modules`）**必须一致**，否则无法加载。

---

## 📝 完整示例脚本

创建 `run_lora_training.sh`：

```bash
#!/bin/bash

# 设置环境变量
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0  # 单卡训练，多卡请使用 torchrun

# 数据集配置
CATEGORY="Industrial_and_Scientific"
TRAIN_FILE="./data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="./data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"
SID_INDEX="./data/Amazon/index/${CATEGORY}.index.json"
ITEM_META="./data/Amazon/index/${CATEGORY}.item.json"

# 模型配置
BASE_MODEL="Qwen/Qwen2-7B-Instruct"

# LoRA 配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET="all"

# 训练配置
BATCH_SIZE=1024
MICRO_BATCH=16
NUM_EPOCHS=10
LEARNING_RATE=3e-4

echo "=========================================="
echo "开始 LoRA 微调训练"
echo "=========================================="
echo "数据集: ${CATEGORY}"
echo "基础模型: ${BASE_MODEL}"
echo "LoRA 配置: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "=========================================="

# SFT 阶段
echo "步骤 1: SFT 训练（LoRA）"
python sft.py \
    --base_model ${BASE_MODEL} \
    --train_file ${TRAIN_FILE} \
    --eval_file ${EVAL_FILE} \
    --output_dir ./output/sft_lora_${CATEGORY} \
    --use_lora True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --category ${CATEGORY} \
    --sid_index_path ${SID_INDEX} \
    --item_meta_path ${ITEM_META} \
    --wandb_project minionerec_lora \
    --wandb_run_name sft_lora_${CATEGORY}

echo "SFT 训练完成！"

# RL 阶段
echo "步骤 2: RL 训练（LoRA）"
python rl.py \
    --model_path ./output/sft_lora_${CATEGORY}/final_checkpoint \
    --train_file ${TRAIN_FILE} \
    --eval_file ${EVAL_FILE} \
    --output_dir ./output/rl_lora_${CATEGORY} \
    --use_lora True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET} \
    --info_file ${INFO_FILE} \
    --sid_index_path ${SID_INDEX} \
    --item_meta_path ${ITEM_META} \
    --category ${CATEGORY} \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_generations 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --reward_type rule \
    --wandb_project minionerec_lora \
    --wandb_run_name rl_lora_${CATEGORY}

echo "RL 训练完成！"
echo "=========================================="
echo "训练完成！模型保存在："
echo "SFT: ./output/sft_lora_${CATEGORY}/final_checkpoint"
echo "RL:  ./output/rl_lora_${CATEGORY}/final_checkpoint"
echo "=========================================="
```

使用：

```bash
chmod +x run_lora_training.sh
./run_lora_training.sh
```

---

## 📊 性能对比（A100）

| 微调方式 | 显存占用 | 训练速度 | 模型大小 |
|---------|---------|---------|---------|
| **全参数微调** | ~28GB | 基准 | 7B 参数 |
| **LoRA (r=16)** | ~12GB | **1.5-2.0x 更快** | ~13M 参数 |

**优势**：
- ✅ 显存占用降低 **57%**
- ✅ 训练速度提升 **50-100%**
- ✅ 模型文件更小（只需保存 LoRA 权重）
- ✅ 性能差距很小（通常 < 3%）

---

## ✅ 检查清单

在开始训练前，确认：

- [ ] 环境已配置（`pip install -r requirements.txt`）
- [ ] PEFT 已安装（`pip install peft`）
- [ ] GPU 可用（`nvidia-smi` 显示 A100）
- [ ] 数据文件存在（`.item.json`, `.index.json`, CSV 文件）
- [ ] 代码已修改（`sft.py` 和 `rl.py` 添加了 LoRA 支持）
- [ ] 基础模型路径正确（HuggingFace 模型 ID 或本地路径）

---

## 🎯 总结

1. **基础模型**：项目中没有，需要从 HuggingFace 下载（代码会自动下载）
2. **LoRA 微调步骤**：
   - 修改 `sft.py` 添加 LoRA 支持
   - 修改 `rl.py` 添加 LoRA 支持
   - 运行 SFT 训练（使用 `--use_lora True`）
   - 运行 RL 训练（使用 `--use_lora True`，且配置与 SFT 一致）
3. **A100 优势**：可以使用更大的模型（7B+）和更大的 batch size

**推荐配置（A100）**：
- 模型：`Qwen/Qwen2.5-7B-Instruct`（最佳性能）或 `Qwen/Qwen2.5-3B-Instruct`（更快训练）
- LoRA: `r=16, alpha=32, dropout=0.05, target_modules="all"`
- Batch size: `1024` (micro_batch_size: `16` for 7B, `32` for 3B)

---

## 🚀 模型已下载后的下一步操作

### 步骤 1：确认模型位置

**检查模型是否在默认位置**：

```bash
# 检查 HuggingFace 缓存目录
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/

# 或者如果下载到了自定义目录，确认路径
# 例如：./models/Qwen2.5-3B-Instruct
```

**模型路径说明**：
- **如果模型在默认位置**（`~/.cache/huggingface/hub/`）：
  - 直接使用模型 ID：`Qwen/Qwen2.5-3B-Instruct`
  - 代码会自动找到模型
- **如果模型在自定义目录**（如 `./models/Qwen2.5-3B-Instruct`）：
  - 使用完整路径：`./models/Qwen2.5-3B-Instruct`
  - 或使用绝对路径：`/path/to/models/Qwen2.5-3B-Instruct`

### 步骤 2：开始 SFT 训练（LoRA）

**方式 1：使用脚本（推荐）**

```bash
# 编辑 run_lora_sft.sh，确认模型路径
# 然后运行
bash run_lora_sft.sh
```

**方式 2：直接运行命令**

**如果模型在默认位置**：

```bash
python sft.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/sft_lora_qwen25_3b \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --batch_size 1024 \
    --micro_batch_size 32 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --wandb_project minionerec_lora \
    --wandb_run_name sft_lora_qwen25_3b
```

**如果模型在自定义目录**：

```bash
python sft.py \
    --base_model ./models/Qwen2.5-3B-Instruct \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/sft_lora_qwen25_3b \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --batch_size 1024 \
    --micro_batch_size 32 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --wandb_project minionerec_lora \
    --wandb_run_name sft_lora_qwen25_3b
```

### 步骤 3：监控训练

**查看 GPU 使用情况**：

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

**预期输出**：

训练开始后，你应该看到：

```
==================================================
启用 LoRA 微调
==================================================
LoRA 目标模块: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
可训练参数: 13,107,200 / 3,000,000,000 (0.44%)
==================================================
Loading index from ./data/Amazon/index/Industrial_and_Scientific.index.json
Adding 765 new tokens to tokenizer
LOAD DATA FINISHED
...
```

### 步骤 4：训练时间估算

对于 `Qwen2.5-3B-Instruct` + LoRA：
- **单卡 A100**：约 **2-4 小时**（取决于数据量）
- **训练速度**：比 7B 模型快约 30-50%

### 步骤 5：训练完成后的 RL 训练

SFT 训练完成后，运行 RL 训练：

```bash
python rl.py \
    --model_path ./output/sft_lora_qwen25_3b/final_checkpoint \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./output/rl_lora_qwen25_3b \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "all" \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --category Industrial_and_Scientific \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_generations 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --reward_type rule \
    --wandb_project minionerec_lora \
    --wandb_run_name rl_lora_qwen25_3b
```

**重要**：RL 阶段的 LoRA 配置（`r`, `alpha`, `target_modules`）**必须与 SFT 阶段一致**！

---

## 📊 步骤 6：评估模型效果

### ⚠️ 重要：LoRA 模型评估方法

由于 LoRA 模型包含扩展的词表（添加了新的 SID tokens），**不能直接使用原始的 `evaluate.py`**。需要使用专门的 `evaluate_lora.py` 脚本。

### 6.1 评估 SFT 模型

**方式 A：使用脚本（推荐）**

```bash
bash evaluate_sft_lora.sh
```

**方式 B：直接运行 Python**

```bash
python evaluate_lora.py \
    --base_model ./models/qwen3b \
    --lora_model ./output/sft_lora_qwen25_3b/final_checkpoint \
    --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --num_beams 50 \
    --K 10
```

**参数说明**：
- `--base_model`：**原始基础模型路径**（如 `./models/qwen3b` 或 `Qwen/Qwen2.5-3B-Instruct`）
- `--lora_model`：**LoRA 模型路径**（如 `./output/sft_lora_qwen25_3b/final_checkpoint`）
- `--test_data_path`：测试数据路径
- `--info_file`：商品信息文件
- `--category`：商品类别
- `--num_beams`：束搜索大小（推荐 50）
- `--K`：Top-K 评估（推荐 10）

### 6.2 评估 RL 模型（RL 训练完成后）

**方式 A：使用脚本（推荐）**

```bash
bash evaluate_rl_lora.sh
```

**方式 B：直接运行 Python**

```bash
python evaluate_lora.py \
    --base_model ./models/qwen3b \
    --lora_model ./output/rl_lora_qwen25_3b/final_checkpoint \
    --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --num_beams 50 \
    --K 10
```

### 6.3 评估指标说明

评估完成后，会输出以下指标：

```
==================================================
评估结果
==================================================
测试样本数: 1000
HR@10: 0.3542
NDCG@10: 0.2156
MRR: 0.1834
==================================================
```

**指标含义**：
- **HR@10 (Hit Rate@10)**：Top-10 推荐中命中目标商品的比例
  - 值越高越好，范围 [0, 1]
  - 例如 0.3542 表示 35.42% 的测试样本在 Top-10 中找到了目标商品

- **NDCG@10 (Normalized Discounted Cumulative Gain@10)**：考虑排名位置的推荐质量
  - 值越高越好，范围 [0, 1]
  - 排名越靠前的命中权重越高

- **MRR (Mean Reciprocal Rank)**：目标商品排名的倒数的平均值
  - 值越高越好，范围 [0, 1]
  - 例如目标商品排第 3，则 RR = 1/3

### 6.4 对比 SFT 和 RL 模型

建议同时评估 SFT 和 RL 模型，对比效果：

```bash
# 1. 评估 SFT 模型
bash evaluate_sft_lora.sh

# 2. 运行 RL 训练
bash rl_lora.sh

# 3. 评估 RL 模型
bash evaluate_rl_lora.sh
```

**预期结果**：
- RL 模型的 HR@10 和 NDCG@10 通常会比 SFT 模型提升 **5-15%**
- 训练时间：SFT (2-4h) + RL (1-2h) = 总共 3-6h

### 6.5 评估结果保存

评估结果会自动保存到：
- SFT 模型：`./output/sft_lora_qwen25_3b/eval_results.json`
- RL 模型：`./output/rl_lora_qwen25_3b/eval_results.json`

可以查看详细结果：
```bash
cat ./output/sft_lora_qwen25_3b/eval_results.json
cat ./output/rl_lora_qwen25_3b/eval_results.json
```

---

## 📈 步骤 7：如何判断训练效果好坏

### 7.1 查看评估结果文件

评估完成后，结果保存在 JSON 文件中。每个测试样本包含：

```bash
cat ./output/sft_lora_qwen25_3b/eval_results.json | head -50
```

**JSON 格式示例**：
```json
[
  {
    "input": "The user has interacted with items <a_123><b_45><c_67>, <a_234><b_56><c_78> in chronological order. Can you predict the next possible item?",
    "output": "<a_145><b_67><c_89>",
    "predict": [
      "<a_145><b_67><c_89>",     // Top-1 预测（正确！）
      "<a_150><b_70><c_90>",     // Top-2 预测
      "<a_140><b_65><c_85>",     // Top-3 预测
      ...
    ]
  },
  ...
]
```

### 7.2 计算评估指标

使用提供的分析脚本计算指标：

```bash
python analyze_results.py \
    --result_file ./output/sft_lora_qwen25_3b/eval_results.json \
    --K 10
```

**输出示例**：
```
==================================================
评估指标分析
==================================================
测试样本数: 1000
Top-K: 10

HR@1:  0.1234  (12.34%)  ← Top-1 命中率
HR@5:  0.2856  (28.56%)  ← Top-5 命中率
HR@10: 0.3542  (35.42%)  ← Top-10 命中率
HR@20: 0.4123  (41.23%)  ← Top-20 命中率

NDCG@5:  0.1823
NDCG@10: 0.2156
NDCG@20: 0.2489

MRR: 0.1834

平均预测排名: 8.45
==================================================
```

### 7.3 评估指标详解

#### 📊 HR@K (Hit Rate @ K) - 命中率

**含义**：在 Top-K 推荐中，目标商品出现的比例

**计算公式**：
```
HR@K = (Top-K 中包含目标商品的样本数) / (总样本数)
```

**判断标准**：
- **HR@10 ≥ 0.30 (30%)**：✅ **良好**
- **HR@10 = 0.20-0.30 (20-30%)**：⚠️ **一般**
- **HR@10 < 0.20 (20%)**：❌ **较差**

**示例**：
- HR@10 = 0.35 表示：35% 的测试样本，目标商品在 Top-10 推荐中
- HR@1 = 0.12 表示：12% 的测试样本，目标商品是第一个推荐

#### 📊 NDCG@K (Normalized Discounted Cumulative Gain @ K) - 排名质量

**含义**：考虑排名位置的推荐质量，排名越靠前权重越高

**计算公式**：
```
DCG@K = Σ (rel_i / log2(i+1))  # rel_i = 1 如果命中，否则 0
NDCG@K = DCG@K / IDCG@K        # 归一化
```

**判断标准**：
- **NDCG@10 ≥ 0.20**：✅ **良好**
- **NDCG@10 = 0.15-0.20**：⚠️ **一般**
- **NDCG@10 < 0.15**：❌ **较差**

**示例**：
- 目标商品排第 1：NDCG 贡献 = 1.0
- 目标商品排第 2：NDCG 贡献 = 0.63
- 目标商品排第 5：NDCG 贡献 = 0.43
- 目标商品排第 10：NDCG 贡献 = 0.30

#### 📊 MRR (Mean Reciprocal Rank) - 平均倒数排名

**含义**：目标商品排名的倒数的平均值

**计算公式**：
```
RR = 1 / rank  # 目标商品的排名
MRR = 平均所有样本的 RR
```

**判断标准**：
- **MRR ≥ 0.18**：✅ **良好**
- **MRR = 0.12-0.18**：⚠️ **一般**
- **MRR < 0.12**：❌ **较差**

**示例**：
- 目标商品排第 1：RR = 1.0
- 目标商品排第 2：RR = 0.5
- 目标商品排第 5：RR = 0.2
- 目标商品排第 10：RR = 0.1

### 7.4 不同数据集的基准性能

#### Amazon Industrial_and_Scientific (本项目默认数据集)

| 模型 | HR@10 | NDCG@10 | MRR |
|------|-------|---------|-----|
| **优秀** | > 0.35 | > 0.22 | > 0.19 |
| **良好** | 0.30-0.35 | 0.18-0.22 | 0.15-0.19 |
| **一般** | 0.25-0.30 | 0.15-0.18 | 0.12-0.15 |
| **较差** | < 0.25 | < 0.15 | < 0.12 |

#### Amazon Office_Products

| 模型 | HR@10 | NDCG@10 | MRR |
|------|-------|---------|-----|
| **优秀** | > 0.40 | > 0.25 | > 0.22 |
| **良好** | 0.35-0.40 | 0.20-0.25 | 0.18-0.22 |
| **一般** | 0.30-0.35 | 0.16-0.20 | 0.14-0.18 |
| **较差** | < 0.30 | < 0.16 | < 0.14 |

**注意**：不同数据集的难度不同，基准性能也不同。

### 7.5 对比 SFT 和 RL 模型

**创建对比脚本 `compare_results.py`**：

```python
import json
import sys

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metrics(results, K=10):
    total = len(results)
    hr_at_k = 0
    ndcg_at_k = 0
    mrr = 0
    
    for item in results:
        target = item['output']
        predictions = item['predict'][:K]
        
        if target in predictions:
            hr_at_k += 1
            rank = predictions.index(target) + 1
            ndcg_at_k += 1.0 / (rank.bit_length())  # log2(rank+1)
            mrr += 1.0 / rank
    
    return {
        'HR@{}'.format(K): hr_at_k / total,
        'NDCG@{}'.format(K): ndcg_at_k / total,
        'MRR': mrr / total
    }

# 加载两个模型的结果
sft_results = load_results('./output/sft_lora_qwen25_3b/eval_results.json')
rl_results = load_results('./output/rl_lora_qwen25_3b/eval_results.json')

# 计算指标
sft_metrics = calculate_metrics(sft_results, K=10)
rl_metrics = calculate_metrics(rl_results, K=10)

# 输出对比
print("=" * 60)
print("SFT vs RL 模型对比")
print("=" * 60)
print(f"{'指标':<15} {'SFT 模型':<15} {'RL 模型':<15} {'提升':<15}")
print("-" * 60)

for key in ['HR@10', 'NDCG@10', 'MRR']:
    sft_val = sft_metrics[key]
    rl_val = rl_metrics[key]
    improvement = ((rl_val - sft_val) / sft_val) * 100
    print(f"{key:<15} {sft_val:<15.4f} {rl_val:<15.4f} {improvement:>+6.2f}%")

print("=" * 60)
```

运行对比：
```bash
python compare_results.py
```

**预期输出**：
```
============================================================
SFT vs RL 模型对比
============================================================
指标             SFT 模型         RL 模型          提升
------------------------------------------------------------
HR@10           0.3200          0.3520          +10.00%
NDCG@10         0.1950          0.2156          +10.56%
MRR             0.1680          0.1834          +9.17%
============================================================
```

**判断标准**：
- **RL 提升 > 8%**：✅ **RL 训练效果显著**
- **RL 提升 3-8%**：⚠️ **RL 训练有一定效果**
- **RL 提升 < 3%**：❌ **RL 训练效果不明显，可能需要调整超参数**
- **RL 提升 < 0%**：❌ **RL 训练失败，需要检查配置**

### 7.6 查看具体预测案例

**查看预测正确的案例**：

```bash
python show_predictions.py --result_file ./output/sft_lora_qwen25_3b/eval_results.json --show_correct --limit 5
```

**查看预测错误的案例**：

```bash
python show_predictions.py --result_file ./output/sft_lora_qwen25_3b/eval_results.json --show_incorrect --limit 5
```

**示例输出**：
```
案例 1 (正确预测):
用户历史: <a_123><b_45><c_67>, <a_234><b_56><c_78>
真实目标: <a_145><b_67><c_89>
Top-5 预测:
  1. <a_145><b_67><c_89> ✅ (正确！排名第 1)
  2. <a_150><b_70><c_90>
  3. <a_140><b_65><c_85>
  4. <a_155><b_72><c_92>
  5. <a_135><b_63><c_83>

案例 2 (错误预测):
用户历史: <a_456><b_78><c_90>, <a_567><b_89><c_01>
真实目标: <a_678><b_90><c_12>
Top-5 预测:
  1. <a_670><b_88><c_10>
  2. <a_680><b_92><c_14>
  3. <a_665><b_86><c_08>
  4. <a_685><b_94><c_16>
  5. <a_660><b_84><c_06>
目标商品排名: 第 15 位 ❌
```

### 7.7 训练效果诊断

#### ✅ 训练效果良好的标志

1. **指标达标**：
   - HR@10 ≥ 0.30
   - NDCG@10 ≥ 0.18
   - MRR ≥ 0.15

2. **RL 提升明显**：
   - RL 比 SFT 提升 > 5%

3. **训练损失收敛**：
   - SFT 训练损失稳定下降
   - RL 训练 reward 稳定上升

4. **预测多样性**：
   - Top-K 预测不重复
   - 预测结果符合语义

#### ❌ 训练效果不佳的标志

1. **指标过低**：
   - HR@10 < 0.20
   - NDCG@10 < 0.12
   - MRR < 0.10

2. **RL 无提升或下降**：
   - RL 比 SFT 提升 < 2%
   - RL 比 SFT 下降

3. **训练不稳定**：
   - 损失震荡
   - 梯度爆炸/消失

4. **预测异常**：
   - 预测结果重复
   - 预测格式错误

#### 🔧 改进建议

**如果 HR@10 < 0.25**：
1. 增加训练轮数：`--num_epochs 15`
2. 增加 LoRA rank：`--lora_r 32`
3. 调整学习率：`--learning_rate 5e-4`
4. 检查数据质量

**如果 RL 无提升**：
1. 调整 beta 参数：`--beta 0.02` 或 `0.08`
2. 增加生成样本数：`--num_generations 32`
3. 调整 RL 学习率：`--learning_rate 5e-7`
4. 增加 RL 训练轮数：`--num_train_epochs 2`

**如果训练不稳定**：
1. 减小学习率：`--learning_rate 1e-4`
2. 增加 warmup：添加 warmup 配置
3. 使用梯度裁剪：检查 `max_grad_norm`
4. 减小 batch size

### 7.8 快速评估脚本

创建 `quick_eval.sh` 用于快速查看关键指标：

```bash
#!/bin/bash

RESULT_FILE=$1

if [ -z "$RESULT_FILE" ]; then
    echo "用法: bash quick_eval.sh <结果文件路径>"
    exit 1
fi

python -c "
import json
import math

with open('$RESULT_FILE', 'r') as f:
    results = json.load(f)

total = len(results)
hr1, hr5, hr10 = 0, 0, 0
ndcg10, mrr = 0, 0

for item in results:
    target = item['output']
    preds = item['predict']
    
    if target in preds[:1]: hr1 += 1
    if target in preds[:5]: hr5 += 1
    if target in preds[:10]: hr10 += 1
    
    if target in preds[:10]:
        rank = preds[:10].index(target) + 1
        ndcg10 += 1.0 / math.log2(rank + 1)
        mrr += 1.0 / rank

print('=' * 50)
print(f'测试样本数: {total}')
print('=' * 50)
print(f'HR@1:  {hr1/total:.4f} ({hr1/total*100:.2f}%)')
print(f'HR@5:  {hr5/total:.4f} ({hr5/total*100:.2f}%)')
print(f'HR@10: {hr10/total:.4f} ({hr10/total*100:.2f}%)')
print(f'NDCG@10: {ndcg10/total:.4f}')
print(f'MRR: {mrr/total:.4f}')
print('=' * 50)

# 判断效果
if hr10/total >= 0.30:
    print('✅ 训练效果：良好')
elif hr10/total >= 0.25:
    print('⚠️  训练效果：一般')
else:
    print('❌ 训练效果：较差，建议调整超参数')
print('=' * 50)
"
```

使用：
```bash
bash quick_eval.sh ./output/sft_lora_qwen25_3b/eval_results.json
bash quick_eval.sh ./output/rl_lora_qwen25_3b/eval_results.json
```

---

## 🎯 完整工作流程总结

```bash
# 1. 环境配置
pip install -r requirements.txt

# 2. 下载模型（如果需要）
# 模型会自动下载，或手动下载到 ./models/qwen3b

# 3. SFT 训练（2-4 小时）
bash sft.sh

# 4. 评估 SFT 模型
bash evaluate_sft_lora.sh

# 5. RL 训练（1-2 小时）
bash rl_lora.sh

# 6. 评估 RL 模型
bash evaluate_rl_lora.sh

# 7. 对比结果
cat ./output/sft_lora_qwen25_3b/eval_results.json
cat ./output/rl_lora_qwen25_3b/eval_results.json
```

---

### 常见问题

**1. 模型路径错误**

错误：`OSError: Can't load config for 'Qwen/Qwen2.5-3B-Instruct'`

解决：
- 确认模型路径正确
- 如果模型在自定义目录，使用完整路径：`./models/Qwen2.5-3B-Instruct`

**2. 显存不足（OOM）**

解决：
- 减小 `batch_size`：从 1024 降到 512
- 减小 `micro_batch_size`：从 32 降到 16
- 减小 `lora_r`：从 16 降到 8

**3. 数据文件找不到**

检查：
```bash
ls -lh ./data/Amazon/train/Industrial_and_Scientific*
ls -lh ./data/Amazon/index/Industrial_and_Scientific*
```

**4. 评估时词表大小不匹配**

错误：
```
RuntimeError: Error(s) in loading state_dict for Qwen2ForCausalLM:
    size mismatch for model.embed_tokens.weight: copying a param with shape torch.Size([152225, 2048]) from checkpoint, the shape in current model is torch.Size([151936, 2048]).
```

解决：
- **不要使用 `evaluate.py`**，使用 `evaluate_lora.py`
- 运行：`bash evaluate_sft_lora.sh` 或 `bash evaluate_rl_lora.sh`
- 原因：LoRA 模型包含扩展的词表，需要先加载基础模型，调整词表大小，再加载 LoRA 适配器

**5. huggingface-hub 版本冲突**

错误：
```
ImportError: huggingface-hub>=0.34.0,<1.0 is required
```

解决：
```bash
pip install "huggingface-hub>=0.34.0,<1.0"
```

**6. peft 模块未安装**

错误：
```
ModuleNotFoundError: No module named 'peft'
```

解决：
```bash
pip install peft
```

祝你训练顺利！🚀


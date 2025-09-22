from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载预训练模型和分词器
def get_model_and_tokenizer(model_name, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # 加载分词器，并启用快速模型
    # 确保pad_token和pad_token_id都设置好
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype="auto"
    ) # 加载适用于序列分类任务的模型
    model.config.pad_token_id = tokenizer.pad_token_id  #同步模型配置与分词器保持一致 
    return model, tokenizer

# 应用LORA适配器，实现参数高效微调
def apply_lora(model, r=8, alpha=16, dropout=0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],  # LLaMA3结构的注意力机制相关模块，目标模块
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS  # 任务类型为序列分类
    )
    model = get_peft_model(model, lora_config)
    return model
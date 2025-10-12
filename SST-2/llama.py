import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from data_deal.dataload import read_tsv, SST2Dataset2
from model.LLM import get_model_and_tokenizer, apply_lora
from train.train import train_one_epoch, eval_model

def save_log(log_file, content):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(content + "\n")

def main():
    # 参数
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    train_file = "./data/train.tsv"
    dev_file = "./data/dev.tsv"
    epochs = 3
    batch_size = 4
    lr = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = "./result/result.txt"

    # 数据
    model, tokenizer = get_model_and_tokenizer(model_name)
    train_data = read_tsv(train_file)
    dev_data = read_tsv(dev_file)
    train_dataset = SST2Dataset2(train_data, tokenizer)
    dev_dataset = SST2Dataset2(dev_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # LoRA
    model = apply_lora(model)
    model.to(device)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs * len(train_loader)
    )

    # 清空日志
    open(log_file, "w").close()

    # 训练
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        acc, f1 = eval_model(model, dev_loader, device)
        log = f"Epoch {epoch+1}: Train loss={train_loss:.4f} | Dev ACC={acc:.4f} | Dev F1={f1:.4f}"
        print(log)
        save_log(log_file, log)

    # 保存模型
    model.save_pretrained("output_lora")
    tokenizer.save_pretrained("output_lora")



if __name__ == "__main__":
    main()

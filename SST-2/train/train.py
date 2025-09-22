import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from transformers import  get_linear_schedule_with_warmup # AdamW,
from torch.optim import AdamW  # 使用PyTorch原生的AdamW
from tqdm import tqdm
import random

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------- 训练、验证、测试 ----------
def train(model, train_loader, val_loader, epochs, lr, device, log_file):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    with open(f"./result/{log_file}.txt", 'w', encoding='utf-8') as f_log:  # 打开日志文件
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs, labels = [x.to(device) for x in batch]
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs[0], labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = np.mean(losses)
            log_str = f"Epoch {epoch+1}: Train loss={train_loss:.4f}\n"
            print(log_str.strip())
            f_log.write(log_str)

            # 验证
            acc, f1 = evaluate(model, val_loader,device )
            val_str = f"Epoch {epoch+1}: Val ACC={acc:.4f}, F1={f1:.4f}\n"
            print(val_str.strip())
            f_log.write(val_str)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"./savemodel/{log_file}.pt")
        f_log.write(f"Best Val ACC: {best_acc}\n")
        print("Best Val ACC:", best_acc)

def train_bert(model, train_loader, val_loader, device, log_file, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_acc = 0
    with open(f"{log_file}.txt", 'w', encoding='utf-8') as f_log:  # 打开日志文件
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for input_ids, mask, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = model(input_ids, attention_mask=mask, labels=label)
                loss = output.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            log_str = f"Epoch {epoch+1}: Train loss={epoch_loss/len(train_loader):.4f}\n"
            print(log_str.strip())
            f_log.write(log_str)


            # 验证
            acc, f1 = evaluate_bert(model, val_loader, device)
            val_str = f"Epoch {epoch+1}: Val ACC={acc:.4f}, F1={f1:.4f}\n"
            print(val_str.strip())
            f_log.write(val_str)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"./savemodel/{log_file}.pt")
                print("Saved best model.")
            f_log.write(f"Best Val ACC: {best_acc}\n")
            print("Best Val ACC:", best_acc)

def evaluate(model, loader,device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = [x.to(device) for x in batch]
            # logits = model(inputs)
            logits, _ = model(inputs)  # 忽略注意力权重
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1

def evaluate_bert(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for input_ids, mask, label in dataloader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            label = label.to(device)
            output = model(input_ids, attention_mask=mask)
            logits = output.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def eval_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        label = batch["labels"].cpu().numpy()
        preds.extend(pred)
        labels.extend(label)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1

# def evaluate(model, loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in loader:
#             inputs, labels = [x.to(DEVICE) for x in batch]
#             logits, _ = model(inputs)  # 忽略注意力权重
#             preds = torch.argmax(logits, dim=1).cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels.cpu().numpy())
#     acc = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds)
#     return acc, f1

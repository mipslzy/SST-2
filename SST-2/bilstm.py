import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.bilstm import BiLSTM, BiLSTMSelfAttention
from train.train import train, evaluate
from data_deal.dataload import read_tsv ,SentimentDataset, load_embeddings


# --------- 参数设置 ----------
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
# LOG_FILE = 'bilstm'
LOG_FILE = 'bilstm_attention'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- 主流程 ----------
def main():
    # 路径配置
    word2vec_path = 'GoogleNews-vectors-negative300.bin'
    train_path = './data/train.tsv'
    val_path = './data/dev.tsv'
    test_path = './data/test.tsv'
    # 加载词向量
    print("Loading word2vec ...")
    word2idx, embeddings = load_embeddings(word2vec_path,EMBEDDING_DIM)

    # 加载数据集
    train_data = read_tsv(train_path)
    val_data = read_tsv(val_path)
    test_data = read_tsv(test_path, test=True)


    train_set = SentimentDataset(train_data, word2idx)
    val_set = SentimentDataset(val_data, word2idx)
    test_set = SentimentDataset(test_data, word2idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # 构建模型
    # model = BiLSTM(embedding_dim=EMBEDDING_DIM,embedding_matrix=embeddings, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model = BiLSTMSelfAttention(embedding_matrix=embeddings, hidden_dim=HIDDEN_DIM).to(DEVICE)
    # 训练
    train(model, train_loader, val_loader, EPOCHS, LR, DEVICE,LOG_FILE)
    model.load_state_dict(torch.load(f"{LOG_FILE}.pt"))

    # 测试
    acc, f1 = evaluate(model, test_loader,DEVICE)
    test_str = f"Test ACC={acc:.4f}, F1={f1:.4f}\n"
    print(test_str.strip())
    # 追加写入测试结果
    with open(f"./result/{LOG_FILE}.txt", 'a', encoding='utf-8') as f_log:
        f_log.write(test_str)

if __name__ == "__main__":
    main()

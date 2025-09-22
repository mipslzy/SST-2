import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import KeyedVectors

from torch.utils.data import Dataset
from transformers import BertTokenizer

# 使用词向量
# --------- 数据处理 ----------
def read_tsv(file, test=False):
    with open(file, encoding="utf-8") as f:
        data = []
        if test:
            lines = f.readlines()
        else :
            lines = f.readlines()[1:] # 跳过表头
        for line in lines:
            if test:
                splits = line.split(' ', 1)  # 只分割第一个空格
                label, sentence = splits
            else:
                splits = line.strip().split('\t')
                sentence, label = splits
            data.append((sentence, int(label)))
        return data

class SentimentDataset(Dataset):
    def __init__(self, data, word2idx, max_len=40):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def encode_sentence(self, sentence):
        tokens = sentence.strip().split()
        idxs = [self.word2idx.get(w.lower(), self.word2idx['<unk>']) for w in tokens]
        if len(idxs) < self.max_len:
            idxs += [self.word2idx['<pad>']] * (self.max_len - len(idxs))
        else:
            idxs = idxs[:self.max_len]
        return np.array(idxs)
    
    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        input_ids = self.encode_sentence(sentence)
        return torch.LongTensor(input_ids), torch.tensor(label, dtype=torch.long)

from torch.utils.data import Dataset
from transformers import BertTokenizer

# --------- 加载词向量 ----------
def load_embeddings(word2vec_path,embedding_dim):
    wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    vocab = ['<pad>', '<unk>'] + list(wv.key_to_index.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    embeddings = np.zeros((len(vocab), embedding_dim))
    embeddings[word2idx['<unk>']] = np.random.randn(embedding_dim)
    for w in wv.key_to_index:
        embeddings[word2idx[w]] = wv[w]
    return word2idx, torch.FloatTensor(embeddings)

# 不使用词向量

class SST2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        # data 是 [(sentence, label), ...]
        self.sentences, self.labels = zip(*data)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.sentences[idx], truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        label = self.labels[idx]
        return input_ids, attention_mask, label

class SST2Dataset2(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        encoded = self.tokenizer(
            sentence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item
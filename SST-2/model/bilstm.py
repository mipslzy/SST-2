import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 注意力模块 ----------
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # BiLSTM输出维度
        self.u = nn.Linear(hidden_dim * 2, 1)

    def forward(self, H):  # H: [batch, seq_len, hidden_dim*2]
        # 计算注意力分数
        scores = self.u(torch.tanh(self.W(H)))  # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)      # [batch, seq_len, 1]
        # 加权求和
        out = torch.sum(weights * H, dim=1)     # [batch, hidden_dim*2]
        return out, weights.squeeze(-1)         # 返回注意力权重可视化

# --------- BiLSTM + Self-Attention模型 ----------
class BiLSTMSelfAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_matrix.size(1), hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                           # [batch, seq_len, embed_dim]
        output, _ = self.bilstm(emb)                      # [batch, seq_len, hidden_dim*2]
        attn_out, attn_weights = self.attention(output)   # [batch, hidden_dim*2], [batch, seq_len]
        logits = self.fc(attn_out)                        # [batch, num_classes]
        return logits, attn_weights

# --------- BiLSTM模型 ----------
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim,embedding_matrix, hidden_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    
    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.bilstm(emb)
        out_pool = torch.mean(output, dim=1)
        logits = self.fc(out_pool)
        return logits

# class BiLSTMWord2Vec(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim=128, output_dim=2, num_layers=1, dropout=0.5):
#         super().__init__()
#         self.bilstm = nn.LSTM(
#             embedding_dim, hidden_dim, num_layers=num_layers,
#             bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
#         )
#         self.fc = nn.Linear(hidden_dim * 2, output_dim) # 双向拼接
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, lengths):
#         # 1.处理变长序列
#         packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         packed_output, (hidden, _) = self.bilstm(packed)

#         # 2.拼接最后一层的双向隐藏状态
#         hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden_dim*2]
#         # batch_size = hidden.size(1)
#         # hidden = hidden.view(self.bilstm.num_layers, 2, batch_size, -1)  # 分开层和方向
#         # last_layer_hidden = hidden[-1]  # 取最后一层，形状: (2, batch_size, hidden_dim)
#         # hidden_cat = torch.cat((last_layer_hidden[0], last_layer_hidden[1]), dim=1)  # 拼接前后向

#         # 3.分类头
#         hidden_cat = self.dropout(hidden_cat)
#         logits = self.fc(hidden_cat)
#         return logits
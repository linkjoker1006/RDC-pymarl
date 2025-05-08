import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert (
            self.head_dim * nhead == d_model
        ), "d_model must be divisible by nhead"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.q_linear(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)

        # Concatenate heads and put through output linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output = self.self_attn(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_output))

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, attn_mask)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        self_attn_output = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, cross_attn_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        ffn_output = self.ffn(x)
        return self.norm3(x + self.dropout3(ffn_output))

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask, cross_attn_mask)

        return self.output_layer(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_dim, d_model, nhead, d_ff, num_layers, dropout)
        self.decoder = Decoder(output_dim, d_model, nhead, d_ff, num_layers, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask)
        return dec_output

# 示例：创建 Transformer 模型
input_dim = 1000  # 输入词汇表大小
output_dim = 1000  # 输出词汇表大小
d_model = 512  # 嵌入维度
nhead = 8  # 多头注意力头数
d_ff = 2048  # 前馈网络的隐藏层维度
num_layers = 6  # 编码器和解码器层数
dropout = 0.1  # dropout 比率

model = Transformer(input_dim, output_dim, d_model, nhead, d_ff, num_layers, dropout)

# 示例输入
src = torch.randint(0, input_dim, (10, 32))  # (源序列长度, 批量大小)
tgt = torch.randint(0, output_dim, (20, 32))  # (目标序列长度, 批量大小)

# 前向传播
output = model(src, tgt)
print(output.shape)  # 输出形状 (目标序列长度, 批量大小, 输出词汇表大小)

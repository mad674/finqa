import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(2, embed_size)
        self.dropout = nn.Dropout(0.1)

        # Create sinusoidal positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))  # (embed_size/2)

        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)  # not a parameter

    def forward(self, input_ids, segment_ids):
        global pbe
        seq_length = input_ids.size(1)

        token_embeds = self.token_embedding(input_ids)  # (B, T, E)
        position_embeds = self.positional_encoding[:seq_length, :].unsqueeze(0)  # (1, T, E)
        segment_embeds = self.segment_embedding(segment_ids)  # (B, T, E)

        embeddings = token_embeds + position_embeds + segment_embeds


        return self.dropout(embeddings)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.scale = math.sqrt(self.head_dim)
        # self.a=0
    def forward(self, x, mask=None):

        global pm
        batch_size, seq_length, embed_size = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)


        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Expands to (batch_size, 1, 1, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
 

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)
   
        return self.fc_out(attention_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_size)
        )

        self.dropout = nn.Dropout(dropout)
   
    def forward(self, x, mask):

        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
     
        feed_forward_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_out))
 
        return x

class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_len=512):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, embed_size, max_len)
        self.encoders = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(embed_size, embed_size)
        self.activation = nn.Tanh()
    def forward(self, input_ids, segment_ids, attention_mask):
    
        x = self.embedding(input_ids, segment_ids)

        for encoder in self.encoders:
            x = encoder(x, attention_mask)
    
        cls_token = x[:, 0]  
        pooled_output = self.activation(self.pooler(cls_token))

        return pooled_output


class BertRetriever(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_layers=12, num_heads=12, hidden_dim=3072, num_labels=2):
        super().__init__()
        self.bert = BERT(vocab_size, embed_size, num_layers, num_heads, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embed_size, num_labels)
        self.a=0
    def forward(self, input_ids, segment_ids, attention_mask):

        pooled_output = self.bert(input_ids, segment_ids, attention_mask)

        pooled_output = self.dropout(pooled_output)
       
        logits = self.classifier(pooled_output)

        return logits

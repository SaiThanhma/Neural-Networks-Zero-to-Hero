import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim, qk_dim, num_heads, context_size = None, dropout = 0.1): # context_size = None means not causal attention
        super().__init__()

        self.query = nn.Linear(in_dim, qk_dim * num_heads, bias=True)
        self.key = nn.Linear(in_dim, qk_dim * num_heads, bias=True)
        self.value = nn.Linear(in_dim, qk_dim * num_heads, bias=True)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.context_size = context_size
        
        if self.context_size is not None:
            self.register_buffer('att_mask', torch.triu(torch.ones((context_size, context_size)), diagonal= 1).bool())

        self.proj = nn.Linear(qk_dim * num_heads, in_dim)

    def forward(self, x):
        B,T, _ = x.shape

        queries = self.query(x) # (B, T, qk_dim * num_heads)
        keys = self.key(x) # (B, T, qk_dim * num_heads)
        values = self.value(x) # (B, T, qk_dim * num_heads)

        queries = queries.reshape(B, T, self.num_heads, -1).transpose(1, 2) # (B, num_heads, T, qk_dim)
        keys = keys.reshape(B, T, self.num_heads, -1).transpose(1, 2) # (B, num_heads, T, qk_dim)
        values = values.reshape(B, T, self.num_heads, -1).transpose(1, 2) # (B, num_heads, T, qk_dim)

        att = queries @ keys.transpose(2, 3) * queries.shape[3]**(-0.5) # (B, num_heads, T, T) = (B, num_heads, T, qk_dim) x (B, num_heads, qk_dim, T)
        if self.context_size is not None:
            att = att.masked_fill(self.att_mask[:T,:T], float("-inf"))
        att_norm = F.softmax(att, dim = 3) # (B, num_heads, T, T)
        att_norm = self.dropout(att_norm)
        v = att_norm @ values #  (B, num_heads, T, qk_dim) = (B, num_heads, T, T) x (B, num_heads, T, qk_dim)
        v = v.transpose(1,2).reshape(B, T, -1)  # (B, T, qk_dim * num_heads)
        out = self.dropout(self.proj(v)) # (B, T, in_dim)
        return out


class CrossAttention(nn.Module):
    def __init__(self, in_enc_dim, in_dec_dim, qk_dim, num_heads, dropout = 0.1):
        super().__init__()
        self.query = nn.Linear(in_dec_dim, qk_dim * num_heads, bias=True) # Check if bias is necessary
        self.key = nn.Linear(in_enc_dim, qk_dim * num_heads, bias=True) # Check if bias is necessary
        self.value = nn.Linear(in_enc_dim, qk_dim * num_heads, bias=True) # Check if bias is necessary
        self.proj = nn.Linear(qk_dim * num_heads, in_dec_dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_dec):
        B,T_dec, _ = x_dec.shape
        B,T_enc, _ = x_enc.shape

        queries = self.query(x_dec) # (B, T, qk_dim * num_heads)
        keys = self.key(x_enc) # (B, T, qk_dim * num_heads)
        values = self.value(x_enc) # (B, T, qk_dim * num_heads)

        queries = queries.reshape(B, T_dec, self.num_heads, -1).transpose(1,2) # (B, num_heads, T_dec, qk_dim)
        keys = keys.reshape(B, T_enc, self.num_heads, -1).transpose(1,2) # (B, num_heads, T_enc, qk_dim)
        values = values.reshape(B, T_enc, self.num_heads, -1).transpose(1,2) # (B, num_heads, T_enc, qk_dim)

        att = queries @ keys.transpose(2, 3) * queries.shape[3]**(-0.5) # (B, num_heads, T_dec, T_enc) = (B, num_heads, T_dec, qk_dim) x (B, num_heads, qk_dim, T_enc)
        att_norm = F.softmax(att, dim = 3) # (B, num_heads, T_dec, T_enc)
        att_norm = self.dropout(att_norm)
        v = att_norm @ values # (B, num_heads, T_dec, qk_dim) = (B, num_heads, T_dec, T_enc) x (B, num_heads, T_enc, qk_dim)
        
        v = v.transpose(1,2).reshape(B, T_dec, -1)  # (B, T_dec, num_heads * qk_dim)
        out = self.dropout(self.proj(v))
        return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        inner_dim = in_dim * 4
        self.first = nn.Linear(in_dim, inner_dim, bias = True)
        self.second = nn.Linear(inner_dim, in_dim, bias = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = self.second(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_dim, qk_dim, num_heads, dropout=0.1):
        super().__init__()
        self.sa = SelfAttention(in_dim, qk_dim, num_heads, dropout = dropout)
        self.ln1 = nn.LayerNorm(in_dim)
        self.ffn = FeedForward(in_dim, dropout = dropout)
        self.ln2 = nn.LayerNorm(in_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_enc_dim, in_dec_dim, qk_dim, num_heads, context_size, dropout = 0.1):
        super().__init__()
        self.sa = SelfAttention(in_dec_dim, qk_dim, num_heads, context_size = context_size, dropout = dropout)
        self.ln1 = nn.LayerNorm(in_dec_dim)
        self.ca = CrossAttention(in_enc_dim, in_dec_dim, qk_dim, num_heads, dropout = dropout)
        self.ln2 = nn.LayerNorm(in_dec_dim)
        self.ffn = FeedForward(in_dec_dim, dropout = dropout)
        self.ln3 = nn.LayerNorm(in_dec_dim)


    def forward(self, x_enc, x_dec):
        x = x_dec + self.sa(self.ln1(x_dec))
        x = x + self.ca(x_enc, self.ln2(x))
        x = x + self.ffn(self.ln3(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_enc_size, emb_enc_dim, context_enc_size, vocab_dec_size, emb_dec_dim, context_dec_size, qk_dim, num_heads, num_layers, dropout = 0.1):
        super().__init__()
        assert num_layers > 0
        self.emb_enc = nn.Embedding(vocab_enc_size, emb_enc_dim)
        self.emb_dec = nn.Embedding(vocab_dec_size, emb_dec_dim)

        self.encoders = nn.ModuleList([Encoder(emb_enc_dim, qk_dim, num_heads,  dropout = dropout) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([Decoder(emb_enc_dim, emb_dec_dim, qk_dim, num_heads, context_dec_size, dropout = dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(emb_dec_dim, vocab_dec_size)

        self.register_buffer('pos_enc_emb', self.positional_encoding(emb_enc_dim, context_enc_size)) 
        self.register_buffer('pos_dec_emb', self.positional_encoding(emb_dec_dim, context_dec_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_dec):
        x_enc_emb = self.dropout(self.emb_enc(x_enc) + self.pos_enc_emb[:x_enc.shape[1]]) # Broadcasting
        x_dec_emb = self.dropout(self.emb_dec(x_dec) + self.pos_dec_emb[:x_dec.shape[1]]) # Broadcasting

        for encoder in self.encoders:
            x_enc_emb = encoder(x_enc_emb)
            
        for decoder in self.decoders:
            x_dec_emb = decoder(x_enc_emb, x_dec_emb)

        out = self.linear(x_dec_emb)
        return out

    def positional_encoding(self, in_dim, length):
        pos = torch.arange(length)[:, None] # (length, 1)
        i = torch.arange(in_dim)[None, :] # (1, in_dim)
        #i = 10000 ** (i / in_dim)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_dim)
        pe = pos * angle_rates  # (length, in_dim)

        #pe = pos / i # (length, in_dim)
        
        pe[:, ::2] = torch.sin(pe[:, ::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe
    
    def generate(self, x_enc, start_token_id, end_token_id, max_length=64):

        self.eval()
        with torch.no_grad():
            current = torch.tensor([[start_token_id]], device=x_enc.device)
            
            for _ in range(max_length):
                logits = self(x_enc, current)  # (1, t, vocab_size)
                logits = logits[:, -1, :]  # (1, vocab_size)
                probs = F.softmax(logits, dim=1)  # (1, vocab_size)
                next_token = torch.multinomial(probs, 1)  # (1, 1)
                current = torch.cat((current, next_token), dim=1)
                
                if next_token.item() == end_token_id:
                    break
                    
        return current
    
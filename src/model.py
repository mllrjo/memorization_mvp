# src/model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (batch_size, sequence_length, d_model)
        # pe is (1, max_len, d_model)
        return x + self.pe[:, :x.size(1)]

class BareBonesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, sequence_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)

        # Transformer Encoder Layer (we'll use this as a building block for simplicity)
        # For a decoder-only, typically TransformerDecoderLayer is used, but for binary prediction
        # and simplicity, a stacked EncoderLayer can sometimes serve as a basic block.
        # We'll adapt it for next-token prediction.
        
        # Using a simple TransformerEncoderLayer for the MVP as a building block.
        # In a true decoder, self-attention would be masked. Here, we mask manually or ensure
        # that the task (predicting next bit) uses a causal mask in training.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer predicts the next bit (0 or 1)
        self.output_head = nn.Linear(d_model, vocab_size) # vocab_size=2 for binary

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # src: (batch_size, sequence_length) - input bit sequence
        
        # Embedding and Positional Encoding
        src = self.embedding(src) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)

        # Generate a causal mask for decoder-like behavior (predicting next token)
        # This mask prevents attention to future tokens.
        # mask is (seq_len, seq_len)
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Pass through Transformer Encoder
        # The TransformerEncoder expects (seq_len, batch_size, d_model) by default if batch_first=False
        # But we set batch_first=True, so it expects (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src, mask=mask) # (batch_size, seq_len, d_model)

        # Output head predicts for each position in the sequence
        # We want to predict the (i+1)-th bit using information up to the i-th bit.
        # The output_head will produce logits for each position.
        logits = self.output_head(output) # (batch_size, seq_len, vocab_size)
        
        return logits

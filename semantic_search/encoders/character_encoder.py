"""
Character Recognition Encoder Module

Handles encoding of recognized characters and OCR results.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class CharacterEncoder(nn.Module):
    """
    Character encoder for processing OCR results and character-level features.
    Encodes character sequences with positional and font information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize character encoder.
        
        Args:
            config: Configuration dictionary containing:
                - char_vocab_size: Size of character vocabulary
                - hidden_dim: Hidden dimension size
                - num_layers: Number of layers
                - num_heads: Number of attention heads
                - max_char_length: Maximum character sequence length
                - embedding_dim: Final embedding dimension
        """
        super(CharacterEncoder, self).__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        self.max_char_length = config.get('max_char_length', 1024)
        
        # Character embedding
        self.char_embedding = nn.Embedding(
            config.get('char_vocab_size', 256),
            self.hidden_dim
        )
        
        # Font/style feature encoder (for font size, style, color, etc.)
        self.style_encoder = nn.Sequential(
            nn.Linear(config.get('style_dim', 8), 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.max_char_length, self.hidden_dim)
        )
        
        # BiLSTM for character sequence modeling
        self.bilstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=config.get('num_layers', 2),
            bidirectional=True,
            batch_first=True,
            dropout=config.get('dropout', 0.1) if config.get('num_layers', 2) > 1 else 0
        )
        
        # Attention mechanism for character importance
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_dim, config.get('embedding_dim', 512))
        
    def forward(
        self,
        char_ids: torch.Tensor,
        style_features: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for character encoding.
        
        Args:
            char_ids: Character IDs [batch_size, char_length]
            style_features: Style features [batch_size, char_length, style_dim]
            attention_mask: Attention mask [batch_size, char_length]
            
        Returns:
            Character embeddings [batch_size, embedding_dim]
        """
        batch_size, char_length = char_ids.shape
        
        # Embed characters
        x = self.char_embedding(char_ids)
        
        # Add style features if provided
        if style_features is not None:
            style_encoded = self.style_encoder(style_features)
            x = x + style_encoded
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :char_length, :]
        
        # Apply BiLSTM
        if attention_mask is not None:
            # Pack padded sequence for efficiency
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            x, _ = self.bilstm(x)
        
        # Self-attention
        attn_mask = None
        if attention_mask is not None:
            attn_mask = ~attention_mask.bool()
            
        residual = x
        x, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.attention_norm(residual + x)
        
        # Pool over character sequence
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)
        
        # Project to final embedding dimension
        embeddings = self.projection(pooled)
        
        return embeddings

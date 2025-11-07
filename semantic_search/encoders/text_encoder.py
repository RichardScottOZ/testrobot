"""
Text Encoder Module

Handles text embedding using transformer-based models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class TextEncoder(nn.Module):
    """
    Text encoder for processing textual content from documents.
    Uses transformer-based architecture for semantic text understanding.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text encoder.
        
        Args:
            config: Configuration dictionary containing:
                - vocab_size: Size of vocabulary
                - hidden_dim: Hidden dimension size
                - num_layers: Number of transformer layers
                - num_heads: Number of attention heads
                - max_seq_length: Maximum sequence length
                - dropout: Dropout rate
        """
        super(TextEncoder, self).__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        self.max_seq_length = config.get('max_seq_length', 512)
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.get('vocab_size', 50000),
            self.hidden_dim
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.get('num_heads', 8),
            dim_feedforward=config.get('ff_dim', 2048),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get('num_layers', 6)
        )
        
        # Projection layer for final embedding
        self.projection = nn.Linear(self.hidden_dim, config.get('embedding_dim', 512))
        
    def forward(self, text_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for text encoding.
        
        Args:
            text_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        batch_size, seq_length = text_ids.shape
        
        # Embed tokens
        x = self.embedding(text_ids)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_length, :]
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            mask = ~attention_mask.bool()
        else:
            mask = None
            
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Pool: use mean pooling over sequence
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

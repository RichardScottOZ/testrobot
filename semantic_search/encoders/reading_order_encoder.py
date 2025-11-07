"""
Reading Order Encoder Module

Handles encoding of reading order and document flow.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ReadingOrderEncoder(nn.Module):
    """
    Reading order encoder for processing document reading sequence.
    Encodes the logical flow and reading order of document elements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reading order encoder.
        
        Args:
            config: Configuration dictionary containing:
                - hidden_dim: Hidden dimension size
                - num_layers: Number of layers
                - num_heads: Number of attention heads
                - max_elements: Maximum number of document elements
                - embedding_dim: Final embedding dimension
        """
        super(ReadingOrderEncoder, self).__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        self.max_elements = config.get('max_elements', 100)
        
        # Element type embedding (paragraph, heading, caption, etc.)
        self.type_embedding = nn.Embedding(
            config.get('num_element_types', 20),
            self.hidden_dim
        )
        
        # Reading order position encoder
        self.order_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
        
        # Hierarchical level encoder (for document structure hierarchy)
        self.level_encoder = nn.Embedding(
            config.get('max_hierarchy_levels', 10),
            self.hidden_dim
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.max_elements, self.hidden_dim)
        )
        
        # Sequential modeling with transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.get('num_heads', 8),
            dim_feedforward=config.get('ff_dim', 2048),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get('num_layers', 4)
        )
        
        # Bidirectional LSTM for capturing reading flow
        self.reading_flow_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_dim, config.get('embedding_dim', 512))
        
    def forward(
        self,
        element_types: torch.Tensor,
        reading_orders: torch.Tensor,
        hierarchy_levels: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for reading order encoding.
        
        Args:
            element_types: Element type IDs [batch_size, num_elements]
            reading_orders: Reading order indices (normalized) [batch_size, num_elements]
            hierarchy_levels: Hierarchy level for each element [batch_size, num_elements]
            mask: Element mask [batch_size, num_elements]
            
        Returns:
            Reading order embeddings [batch_size, embedding_dim]
        """
        batch_size, num_elements = element_types.shape
        
        # Encode element types
        x = self.type_embedding(element_types)
        
        # Add reading order information
        order_encoded = self.order_encoder(reading_orders.unsqueeze(-1))
        x = x + order_encoded
        
        # Add hierarchy level information if provided
        if hierarchy_levels is not None:
            level_encoded = self.level_encoder(hierarchy_levels)
            x = x + level_encoded
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :num_elements, :]
        
        # Apply transformer for global context
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
            
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Apply BiLSTM for sequential reading flow
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.reading_flow_lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            x, _ = self.reading_flow_lstm(x)
        
        # Pool over elements
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)
        
        # Project to final embedding dimension
        embeddings = self.projection(pooled)
        
        return embeddings

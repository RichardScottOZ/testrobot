"""
Panel Layout Encoder Module

Handles encoding of document panel/layout structure.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class PanelEncoder(nn.Module):
    """
    Panel encoder for processing spatial layout and panel structure.
    Encodes bounding boxes and spatial relationships between document panels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize panel encoder.
        
        Args:
            config: Configuration dictionary containing:
                - hidden_dim: Hidden dimension size
                - num_layers: Number of layers
                - max_panels: Maximum number of panels
                - embedding_dim: Final embedding dimension
        """
        super(PanelEncoder, self).__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        self.max_panels = config.get('max_panels', 50)
        
        # Bounding box encoder (x, y, width, height, area)
        self.bbox_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim)
        )
        
        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Positional encoding for panel order
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.max_panels, self.hidden_dim)
        )
        
        # Graph attention for panel relationships
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.get('num_heads', 8),
                dropout=config.get('dropout', 0.1),
                batch_first=True
            )
            for _ in range(config.get('num_layers', 3))
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(config.get('num_layers', 3))
        ])
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_dim, config.get('embedding_dim', 512))
        
    def forward(self, bboxes: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for panel encoding.
        
        Args:
            bboxes: Bounding boxes [batch_size, num_panels, 4] (x, y, w, h)
            mask: Panel mask [batch_size, num_panels]
            
        Returns:
            Panel layout embeddings [batch_size, embedding_dim]
        """
        batch_size, num_panels, _ = bboxes.shape
        
        # Compute bounding box features (add area as 5th feature)
        areas = (bboxes[..., 2] * bboxes[..., 3]).unsqueeze(-1)
        bbox_features = torch.cat([bboxes, areas], dim=-1)
        
        # Encode bounding boxes
        x = self.bbox_encoder(bbox_features)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :num_panels, :]
        
        # Apply attention layers for capturing spatial relationships
        for attn, norm in zip(self.attention_layers, self.norm_layers):
            # Self-attention
            attn_mask = None
            if mask is not None:
                # Create attention mask
                attn_mask = ~mask.bool()
                
            residual = x
            x, _ = attn(x, x, x, key_padding_mask=attn_mask)
            x = norm(residual + x)
        
        # Pool over panels
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

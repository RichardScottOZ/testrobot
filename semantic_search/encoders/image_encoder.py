"""
Image Encoder Module

Handles image embedding using CNN and vision transformer architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ImageEncoder(nn.Module):
    """
    Image encoder for processing visual content from documents.
    Uses CNN backbone followed by vision transformer for feature extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image encoder.
        
        Args:
            config: Configuration dictionary containing:
                - input_channels: Number of input channels (3 for RGB)
                - hidden_dim: Hidden dimension size
                - num_layers: Number of transformer layers
                - num_heads: Number of attention heads
                - patch_size: Size of image patches
                - image_size: Input image size
                - embedding_dim: Final embedding dimension
        """
        super(ImageEncoder, self).__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        self.patch_size = config.get('patch_size', 16)
        self.image_size = config.get('image_size', 224)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # CNN backbone for initial feature extraction
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(config.get('input_channels', 3), 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            256, self.hidden_dim,
            kernel_size=self.patch_size // 4,
            stride=self.patch_size // 4
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.hidden_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        # Vision transformer
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
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_dim, config.get('embedding_dim', 512))
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image encoding.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Image embeddings [batch_size, embedding_dim]
        """
        batch_size = images.shape[0]
        
        # Extract features with CNN
        x = self.cnn_backbone(images)
        
        # Create patches
        x = self.patch_embedding(x)  # [batch_size, hidden_dim, h', w']
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use CLS token output
        cls_output = x[:, 0]
        
        # Project to final embedding dimension
        embeddings = self.projection(cls_output)
        
        return embeddings

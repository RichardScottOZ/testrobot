"""
Multimodal Semantic Model

Main model integrating all encoders with fusion mechanism.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..encoders import (
    TextEncoder,
    ImageEncoder,
    PanelEncoder,
    CharacterEncoder,
    ReadingOrderEncoder
)


class MultimodalSemanticModel(nn.Module):
    """
    Multimodal semantic model for document understanding.
    Integrates text, images, panels, characters, and reading order.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multimodal semantic model.
        
        Args:
            config: Configuration dictionary containing encoder configs and fusion settings
        """
        super(MultimodalSemanticModel, self).__init__()
        
        self.embedding_dim = config.get('embedding_dim', 512)
        self.use_text = config.get('use_text', True)
        self.use_image = config.get('use_image', True)
        self.use_panel = config.get('use_panel', True)
        self.use_character = config.get('use_character', True)
        self.use_reading_order = config.get('use_reading_order', True)
        
        # Initialize encoders
        if self.use_text:
            self.text_encoder = TextEncoder(config.get('text_encoder', {}))
        
        if self.use_image:
            self.image_encoder = ImageEncoder(config.get('image_encoder', {}))
        
        if self.use_panel:
            self.panel_encoder = PanelEncoder(config.get('panel_encoder', {}))
        
        if self.use_character:
            self.character_encoder = CharacterEncoder(config.get('character_encoder', {}))
        
        if self.use_reading_order:
            self.reading_order_encoder = ReadingOrderEncoder(config.get('reading_order_encoder', {}))
        
        # Fusion mechanism
        self.fusion_type = config.get('fusion_type', 'attention')  # 'concat', 'attention', 'gated'
        
        if self.fusion_type == 'concat':
            self._init_concat_fusion(config)
        elif self.fusion_type == 'attention':
            self._init_attention_fusion(config)
        elif self.fusion_type == 'gated':
            self._init_gated_fusion(config)
        
        # Final projection
        self.final_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.final_norm = nn.LayerNorm(self.embedding_dim)
        
    def _init_concat_fusion(self, config):
        """Initialize concatenation-based fusion."""
        num_modalities = sum([
            self.use_text, self.use_image, self.use_panel,
            self.use_character, self.use_reading_order
        ])
        self.fusion_projection = nn.Sequential(
            nn.Linear(self.embedding_dim * num_modalities, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )
    
    def _init_attention_fusion(self, config):
        """Initialize attention-based fusion."""
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.get('fusion_heads', 8),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        
        # Learnable query for fusion
        self.fusion_query = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
    def _init_gated_fusion(self, config):
        """Initialize gated fusion mechanism."""
        num_modalities = sum([
            self.use_text, self.use_image, self.use_panel,
            self.use_character, self.use_reading_order
        ])
        
        # Gate networks for each modality
        self.modality_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])
        
    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        bbox_mask: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
        char_mask: Optional[torch.Tensor] = None,
        char_styles: Optional[torch.Tensor] = None,
        element_types: Optional[torch.Tensor] = None,
        reading_orders: Optional[torch.Tensor] = None,
        hierarchy_levels: Optional[torch.Tensor] = None,
        reading_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multimodal model.
        
        Args:
            text_ids: Text token IDs
            text_mask: Text attention mask
            images: Image tensors
            bboxes: Bounding boxes for panels
            bbox_mask: Panel mask
            char_ids: Character IDs
            char_mask: Character mask
            char_styles: Character style features
            element_types: Document element types
            reading_orders: Reading order indices
            hierarchy_levels: Hierarchy levels
            reading_mask: Reading order mask
            
        Returns:
            Multimodal embeddings [batch_size, embedding_dim]
        """
        modality_embeddings = []
        
        # Encode each modality
        if self.use_text and text_ids is not None:
            text_emb = self.text_encoder(text_ids, text_mask)
            modality_embeddings.append(text_emb)
        
        if self.use_image and images is not None:
            image_emb = self.image_encoder(images)
            modality_embeddings.append(image_emb)
        
        if self.use_panel and bboxes is not None:
            panel_emb = self.panel_encoder(bboxes, bbox_mask)
            modality_embeddings.append(panel_emb)
        
        if self.use_character and char_ids is not None:
            char_emb = self.character_encoder(char_ids, char_styles, char_mask)
            modality_embeddings.append(char_emb)
        
        if self.use_reading_order and element_types is not None:
            reading_emb = self.reading_order_encoder(
                element_types, reading_orders, hierarchy_levels, reading_mask
            )
            modality_embeddings.append(reading_emb)
        
        # Fuse modalities
        if len(modality_embeddings) == 0:
            raise ValueError("At least one modality must be provided")
        
        if len(modality_embeddings) == 1:
            fused = modality_embeddings[0]
        else:
            fused = self._fuse_modalities(modality_embeddings)
        
        # Final projection and normalization
        output = self.final_projection(fused)
        output = self.final_norm(output)
        
        return output
    
    def _fuse_modalities(self, embeddings):
        """Fuse multiple modality embeddings."""
        if self.fusion_type == 'concat':
            # Concatenate and project
            concatenated = torch.cat(embeddings, dim=-1)
            fused = self.fusion_projection(concatenated)
            
        elif self.fusion_type == 'attention':
            # Stack embeddings
            stacked = torch.stack(embeddings, dim=1)  # [batch_size, num_modalities, embedding_dim]
            
            # Use learnable query for attention-based fusion
            batch_size = stacked.size(0)
            query = self.fusion_query.expand(batch_size, -1, -1)
            
            # Apply multi-head attention
            attended, _ = self.modality_attention(query, stacked, stacked)
            fused = self.attention_norm(attended.squeeze(1))
            
        elif self.fusion_type == 'gated':
            # Gated fusion
            gates = []
            for i, (emb, gate_net) in enumerate(zip(embeddings, self.modality_gates)):
                gate = gate_net(emb)
                gates.append(gate)
            
            # Normalize gates
            gates = torch.stack(gates, dim=1)  # [batch_size, num_modalities, 1]
            gates = torch.softmax(gates, dim=1)
            
            # Weight and sum embeddings
            stacked = torch.stack(embeddings, dim=1)  # [batch_size, num_modalities, embedding_dim]
            fused = (stacked * gates).sum(dim=1)
        
        return fused
    
    def encode_query(self, query_text: torch.Tensor, query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a text query for semantic search.
        
        Args:
            query_text: Query text token IDs
            query_mask: Query attention mask
            
        Returns:
            Query embeddings
        """
        if not self.use_text:
            raise ValueError("Text encoder must be enabled for query encoding")
        
        query_emb = self.text_encoder(query_text, query_mask)
        query_emb = self.final_projection(query_emb)
        query_emb = self.final_norm(query_emb)
        
        return query_emb

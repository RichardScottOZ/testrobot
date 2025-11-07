"""
Semantic Search Framework for Multimodal Document Understanding

This package provides a deep learning framework for integrating text, images,
panels, characters, and reading order into semantic search capabilities.
"""

__version__ = "0.1.0"

from .models.multimodal_semantic_model import MultimodalSemanticModel
from .search_engine import SemanticSearchEngine

__all__ = [
    "MultimodalSemanticModel",
    "SemanticSearchEngine",
]

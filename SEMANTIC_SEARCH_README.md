# Multimodal Semantic Search Framework

A deep learning framework for integrating **text**, **images**, **panels**, **characters**, and **reading order** into semantic search capabilities. This framework enables powerful document understanding and retrieval based on multimodal content.

## Features

- **Text Encoding**: Transformer-based text understanding with positional encoding
- **Image Encoding**: CNN + Vision Transformer for visual content processing
- **Panel Layout Encoding**: Spatial relationship understanding with graph attention
- **Character Recognition Encoding**: OCR integration with character-level features
- **Reading Order Encoding**: Document flow and hierarchy understanding
- **Flexible Fusion**: Multiple fusion strategies (concatenation, attention, gated)
- **Efficient Search**: FAISS-based similarity search with multiple index types
- **Modular Design**: Use all or some modalities based on your needs

## Architecture

The framework consists of five main encoder modules:

1. **TextEncoder**: Processes textual content using transformer architecture
2. **ImageEncoder**: Extracts visual features using CNN backbone + ViT
3. **PanelEncoder**: Encodes spatial layout and panel relationships
4. **CharacterEncoder**: Processes character-level OCR results with style features
5. **ReadingOrderEncoder**: Captures document structure and reading flow

These encoders are integrated through a **MultimodalSemanticModel** that fuses the representations using configurable fusion mechanisms.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- faiss-cpu (or faiss-gpu for GPU acceleration)
- transformers

## Quick Start

### 1. Prepare Your Data

Create a JSON file with your documents:

```json
[
  {
    "text": "Document text content",
    "image": "path/to/image.jpg",
    "bboxes": [[10, 20, 100, 50], [15, 80, 120, 60]],
    "image_width": 800,
    "image_height": 600,
    "ocr_text": "Recognized character text",
    "elements": [
      {"type": "heading", "order": 0, "level": 0},
      {"type": "paragraph", "order": 1, "level": 1}
    ]
  }
]
```

### 2. Train the Model

```bash
python train.py \
  --train-data data/train_documents.json \
  --val-data data/val_documents.json \
  --output-dir checkpoints/ \
  --save-interval 10
```

### 3. Build Search Index

```bash
python inference.py \
  --model-path checkpoints/best_model.pt \
  --documents data/corpus_documents.json \
  --save-index indexes/corpus.index
```

### 4. Search

Interactive mode:
```bash
python inference.py \
  --model-path checkpoints/best_model.pt \
  --load-index indexes/corpus.index \
  --interactive
```

Single query:
```bash
python inference.py \
  --model-path checkpoints/best_model.pt \
  --load-index indexes/corpus.index \
  --query "your search query" \
  --top-k 5 \
  --output results.json
```

## Usage Examples

### Using the Framework Programmatically

```python
import torch
from semantic_search import MultimodalSemanticModel, SemanticSearchEngine
from semantic_search.utils import DocumentPreprocessor
from semantic_search.config import get_config

# Initialize model
config = get_config()
model = MultimodalSemanticModel(config)

# Initialize search engine
search_engine = SemanticSearchEngine(
    model=model,
    embedding_dim=512,
    index_type='flat'
)

# Prepare documents
preprocessor = DocumentPreprocessor(config['preprocessing'])
documents = [
    {
        'text': 'Sample document text',
        'image': 'path/to/image.jpg',
        'bboxes': [[10, 20, 100, 50]],
        'image_width': 800,
        'image_height': 600
    }
]

# Preprocess and add documents
preprocessed_docs = [preprocessor.preprocess_document(doc) for doc in documents]
search_engine.add_documents(preprocessed_docs)

# Search
query_data = preprocessor.preprocess_document({'text': 'search query'})
results = search_engine.search_by_document(query_data, top_k=5)

for result in results:
    print(f"Rank {result['rank']}: Score {result['score']:.4f}")
```

### Custom Configuration

```python
from semantic_search.config import get_config

# Create custom configuration
custom_config = {
    'embedding_dim': 768,
    'fusion_type': 'attention',
    'text_encoder': {
        'hidden_dim': 768,
        'num_layers': 12,
    }
}

config = get_config(custom_config)
model = MultimodalSemanticModel(config)
```

## Configuration

The framework uses a hierarchical configuration system. Key configuration options:

### Model Configuration
- `embedding_dim`: Final embedding dimension (default: 512)
- `fusion_type`: Fusion mechanism - 'concat', 'attention', or 'gated'
- `use_text`, `use_image`, `use_panel`, `use_character`, `use_reading_order`: Enable/disable modalities

### Encoder Configurations
Each encoder has its own configuration for:
- `hidden_dim`: Hidden layer dimensions
- `num_layers`: Number of transformer/network layers
- `num_heads`: Number of attention heads
- `dropout`: Dropout rate

### Training Configuration
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimizer
- `num_epochs`: Number of training epochs
- `weight_decay`: L2 regularization weight

## Model Architecture Details

### Text Encoder
- Embedding layer with positional encoding
- Multi-layer transformer encoder
- Mean pooling over sequence
- Projection to embedding dimension

### Image Encoder
- CNN backbone for feature extraction
- Patch embedding
- Vision transformer with CLS token
- Projection layer

### Panel Encoder
- Bounding box feature encoding
- Graph attention for spatial relationships
- Multi-head attention layers
- Pooling and projection

### Character Encoder
- Character-level embedding
- Style feature encoding (font, size, color)
- BiLSTM for sequence modeling
- Multi-head attention
- Pooling and projection

### Reading Order Encoder
- Element type embedding
- Reading order position encoding
- Hierarchy level encoding
- Transformer + BiLSTM
- Pooling and projection

### Fusion Mechanisms

**Concatenation Fusion**: Concatenate all modality embeddings and project

**Attention Fusion**: Use multi-head attention with learnable query to fuse modalities

**Gated Fusion**: Learn importance weights for each modality and compute weighted sum

## Search Index Types

- **flat**: Exact search (slower but accurate)
- **ivf**: Inverted file index (faster approximate search)
- **hnsw**: Hierarchical navigable small world graph (very fast approximate search)

## Training

The framework supports training with:
- **Contrastive Loss**: For learning discriminative embeddings
- **Triplet Loss**: For metric learning with anchor-positive-negative triplets

Custom loss functions can be easily integrated.

## Performance Considerations

- Use GPU for faster training and inference
- Choose appropriate FAISS index type based on corpus size
- Adjust batch size based on available memory
- Consider using mixed precision training for large models

## Advanced Features

### Custom Encoders
You can replace any encoder with your own implementation:

```python
from semantic_search.encoders import TextEncoder

class CustomTextEncoder(TextEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers
    
    def forward(self, text_ids, attention_mask=None):
        # Custom forward pass
        pass
```

### Custom Fusion
Implement custom fusion strategies by extending the model:

```python
def _fuse_modalities(self, embeddings):
    # Custom fusion logic
    return fused_embedding
```

## Citation

If you use this framework in your research, please cite:

```
@software{multimodal_semantic_search,
  title={Multimodal Semantic Search Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/RichardScottOZ/testrobot}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.

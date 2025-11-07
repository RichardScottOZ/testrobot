# Architecture Guide

## System Overview

The Multimodal Semantic Search Framework is designed with a modular architecture that allows flexible integration of different modalities (text, images, panels, characters, reading order) into a unified semantic embedding space.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document Input                                │
│  (Text, Images, Panels, Characters, Reading Order)               │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Document Preprocessing                          │
│  • Text Tokenization  • Image Normalization                      │
│  • BBox Normalization • Character Encoding                       │
│  • Reading Order Extraction                                      │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Modality Encoders                             │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Text    │  │  Image   │  │  Panel   │  │Character │        │
│  │ Encoder  │  │ Encoder  │  │ Encoder  │  │ Encoder  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
│       ▼             ▼             ▼             ▼                │
│  ┌─────────────────────────────────────────────────┐            │
│  │       Reading Order Encoder                      │            │
│  └─────────────────┬───────────────────────────────┘            │
│                    │                                              │
└────────────────────┼──────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Fusion Mechanism                              │
│                                                                   │
│  Attention Fusion:   [Query] ◄─── [Modality Embeddings]         │
│  Concat Fusion:      [Concat all] → [MLP]                       │
│  Gated Fusion:       [Gate Networks] → [Weighted Sum]           │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│               Final Embedding (512-dim)                          │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FAISS Search Index                               │
│  • Flat (Exact Search)                                           │
│  • IVF (Inverted File)                                           │
│  • HNSW (Hierarchical Navigable Small World)                     │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Similarity Search Results                           │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Text Encoder

**Architecture**: Transformer-based encoder

```
Input Text → Tokenization → Embedding Layer
           ↓
     Positional Encoding
           ↓
     Transformer Layers (6x)
           ↓
     Mean Pooling
           ↓
     Projection Layer → Text Embedding (512-dim)
```

**Features**:
- Word-level tokenization
- Learned positional encodings
- Multi-head self-attention (8 heads)
- Layer normalization
- Dropout for regularization

### 2. Image Encoder

**Architecture**: CNN + Vision Transformer

```
Image Input → CNN Backbone (ResNet-style)
           ↓
     Patch Embedding
           ↓
     CLS Token + Positional Encoding
           ↓
     Transformer Layers (6x)
           ↓
     CLS Token Output
           ↓
     Projection Layer → Image Embedding (512-dim)
```

**Features**:
- CNN feature extraction
- Patch-based representation
- Vision transformer processing
- CLS token aggregation

### 3. Panel Encoder

**Architecture**: Graph Attention Network

```
Bounding Boxes → Feature Encoding (x, y, w, h, area)
              ↓
        Positional Encoding
              ↓
        Multi-head Attention Layers (3x)
              ↓
        Mean Pooling
              ↓
        Projection Layer → Panel Embedding (512-dim)
```

**Features**:
- Spatial relationship modeling
- Graph-based attention
- Hierarchical layout understanding

### 4. Character Encoder

**Architecture**: BiLSTM + Attention

```
Characters → Character Embedding + Style Features
          ↓
    Positional Encoding
          ↓
    Bidirectional LSTM (2 layers)
          ↓
    Multi-head Attention
          ↓
    Mean Pooling
          ↓
    Projection Layer → Character Embedding (512-dim)
```

**Features**:
- Character-level encoding
- Font/style integration
- Bidirectional context
- Attention mechanism

### 5. Reading Order Encoder

**Architecture**: Transformer + BiLSTM

```
Elements → Type Embedding + Order Encoding + Hierarchy
        ↓
   Positional Encoding
        ↓
   Transformer Layers (4x)
        ↓
   Bidirectional LSTM (2 layers)
        ↓
   Mean Pooling
        ↓
   Projection Layer → Reading Order Embedding (512-dim)
```

**Features**:
- Element type encoding
- Reading flow modeling
- Document hierarchy
- Sequential dependencies

### 6. Fusion Mechanisms

#### Attention Fusion
```
Modality Embeddings → Stack
                    ↓
              Learnable Query
                    ↓
           Multi-head Attention
                    ↓
           Fused Embedding
```

#### Concatenation Fusion
```
Modality Embeddings → Concatenate
                    ↓
                  MLP (2 layers)
                    ↓
              Fused Embedding
```

#### Gated Fusion
```
Modality Embeddings → Gate Networks (per modality)
                    ↓
                  Softmax
                    ↓
              Weighted Sum
                    ↓
              Fused Embedding
```

## Training Pipeline

```
Documents → Preprocessing
         ↓
    DataLoader (batching)
         ↓
    Model Forward Pass
         ↓
    Loss Computation (Contrastive/Triplet)
         ↓
    Backward Pass
         ↓
    Optimizer Step
         ↓
    Repeat for Epochs
```

## Inference Pipeline

```
Query → Preprocessing
      ↓
   Model Encoding
      ↓
   FAISS Search
      ↓
   Ranked Results
```

## Configuration System

The framework uses a hierarchical configuration system:

```
DEFAULT_CONFIG
├── Model Settings
│   ├── embedding_dim
│   ├── fusion_type
│   └── use_* flags
├── Encoder Configs
│   ├── text_encoder
│   ├── image_encoder
│   ├── panel_encoder
│   ├── character_encoder
│   └── reading_order_encoder
├── Training Settings
│   ├── batch_size
│   ├── learning_rate
│   └── num_epochs
└── Preprocessing Settings
    ├── image_size
    ├── max_text_length
    └── max_panels
```

## Performance Considerations

### Memory Optimization
- Gradient checkpointing for large models
- Mixed precision training (FP16)
- Batch size adjustment based on GPU memory

### Speed Optimization
- FAISS GPU indices for faster search
- Efficient data loading with multiple workers
- Model quantization for deployment

### Scalability
- Distributed training support
- Index sharding for large corpora
- Incremental index updates

## Extension Points

1. **Custom Encoders**: Implement new encoder classes
2. **Custom Fusion**: Add fusion strategies
3. **Custom Losses**: Define task-specific losses
4. **Custom Preprocessing**: Extend preprocessor class
5. **Custom Search**: Implement domain-specific retrieval

## Design Principles

- **Modularity**: Each component is independent
- **Flexibility**: Easy to enable/disable modalities
- **Extensibility**: Simple to add new features
- **Efficiency**: Optimized for large-scale search
- **Maintainability**: Clear code structure and documentation

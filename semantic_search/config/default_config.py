"""
Default Configuration for Multimodal Semantic Search
"""

DEFAULT_CONFIG = {
    # Model settings
    'embedding_dim': 512,
    'use_text': True,
    'use_image': True,
    'use_panel': True,
    'use_character': True,
    'use_reading_order': True,
    
    # Fusion settings
    'fusion_type': 'attention',  # 'concat', 'attention', 'gated'
    'fusion_heads': 8,
    'dropout': 0.1,
    
    # Text encoder config
    'text_encoder': {
        'vocab_size': 50000,
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ff_dim': 2048,
        'max_seq_length': 512,
        'dropout': 0.1,
        'embedding_dim': 512,
    },
    
    # Image encoder config
    'image_encoder': {
        'input_channels': 3,
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ff_dim': 2048,
        'patch_size': 16,
        'image_size': 224,
        'dropout': 0.1,
        'embedding_dim': 512,
    },
    
    # Panel encoder config
    'panel_encoder': {
        'hidden_dim': 512,
        'num_layers': 3,
        'num_heads': 8,
        'max_panels': 50,
        'dropout': 0.1,
        'embedding_dim': 512,
    },
    
    # Character encoder config
    'character_encoder': {
        'char_vocab_size': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'num_heads': 8,
        'max_char_length': 1024,
        'style_dim': 8,
        'dropout': 0.1,
        'embedding_dim': 512,
    },
    
    # Reading order encoder config
    'reading_order_encoder': {
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 2048,
        'max_elements': 100,
        'num_element_types': 20,
        'max_hierarchy_levels': 10,
        'dropout': 0.1,
        'embedding_dim': 512,
    },
    
    # Training settings
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
    },
    
    # Search settings
    'search': {
        'index_type': 'flat',  # 'flat', 'ivf', 'hnsw'
        'top_k': 10,
    },
    
    # Preprocessing settings
    'preprocessing': {
        'image_size': 224,
        'max_text_length': 512,
        'max_char_length': 1024,
        'max_panels': 50,
        'max_elements': 100,
    }
}


def get_config(config_overrides=None):
    """
    Get configuration with optional overrides.
    
    Args:
        config_overrides: Dictionary of config overrides
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_overrides:
        # Deep merge
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, config_overrides)
    
    return config

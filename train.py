"""
Training Script for Multimodal Semantic Search Model

This script demonstrates how to train the multimodal semantic model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import os

from semantic_search.models import MultimodalSemanticModel
from semantic_search.utils import (
    DocumentPreprocessor,
    ContrastiveLoss,
    MultimodalDataset,
    Trainer,
    collate_multimodal
)
from semantic_search.config import get_config


def load_documents(data_path):
    """
    Load documents from JSON file.
    
    Args:
        data_path: Path to JSON file with documents
        
    Returns:
        List of documents
    """
    with open(data_path, 'r') as f:
        documents = json.load(f)
    return documents


def main(args):
    """Main training function."""
    
    # Load configuration
    config = get_config()
    if args.config:
        with open(args.config, 'r') as f:
            config_overrides = json.load(f)
        config = get_config(config_overrides)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalSemanticModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load data
    print(f"Loading training data from {args.train_data}...")
    train_documents = load_documents(args.train_data)
    print(f"Loaded {len(train_documents)} training documents")
    
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_documents = load_documents(args.val_data)
        print(f"Loaded {len(val_documents)} validation documents")
    else:
        val_documents = []
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(config['preprocessing'])
    
    # Create datasets
    train_dataset = MultimodalDataset(train_documents, preprocessor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=args.num_workers
    )
    
    val_loader = None
    if val_documents:
        val_dataset = MultimodalDataset(val_documents, preprocessor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_multimodal,
            num_workers=args.num_workers
        )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize loss function
    loss_fn = ContrastiveLoss(temperature=0.07)
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, loss_fn, device)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss = trainer.evaluate(val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, save_path)
                print(f"Saved best model to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multimodal Semantic Search Model')
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation data JSON file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

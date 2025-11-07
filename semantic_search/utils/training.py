"""
Training Utilities
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multimodal learning."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [batch_size, embedding_dim]
            embeddings2: Second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings1 = nn.functional.normalize(embeddings1, dim=1)
        embeddings2 = nn.functional.normalize(embeddings2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = embeddings1.size(0)
        labels = torch.arange(batch_size).to(embeddings1.device)
        
        # Compute loss in both directions
        loss1 = self.criterion(similarity, labels)
        loss2 = self.criterion(similarity.T, labels)
        
        return (loss1 + loss2) / 2


class TripletLoss(nn.Module):
    """Triplet loss for embedding learning."""
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_distance = torch.sum((anchor - positive) ** 2, dim=1)
        neg_distance = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute loss
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        return loss.mean()


class MultimodalDataset(Dataset):
    """Dataset for multimodal documents."""
    
    def __init__(self, documents: list, preprocessor):
        """
        Initialize dataset.
        
        Args:
            documents: List of document dictionaries
            preprocessor: Document preprocessor instance
        """
        self.documents = documents
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        document = self.documents[idx]
        return self.preprocessor.preprocess_document(document)


def collate_multimodal(batch):
    """
    Collate function for multimodal data.
    
    Args:
        batch: List of preprocessed documents
        
    Returns:
        Batched dictionary
    """
    batched = {}
    
    # Get all keys from first item
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch if key in item]
        if values:
            # Stack or pad as needed
            if values[0].dim() == 0:
                # Scalar
                batched[key] = torch.stack(values)
            elif values[0].dim() == 1:
                # 1D tensor - pad
                max_len = max(v.size(0) for v in values)
                padded = []
                for v in values:
                    if v.size(0) < max_len:
                        padding = torch.zeros(max_len - v.size(0), dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded.append(v)
                batched[key] = torch.stack(padded)
            else:
                # Multi-dimensional - assume already properly sized or handle case by case
                try:
                    batched[key] = torch.cat(values, dim=0)
                except (RuntimeError, ValueError):
                    batched[key] = torch.stack(values)
    
    return batched


class Trainer:
    """Trainer for multimodal semantic model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(**batch)
            
            # Compute loss
            # NOTE: This is a simplified training loop. In production:
            # - Use proper positive/negative sampling
            # - Create augmented views or use labeled pairs
            # - Implement batch construction for contrastive learning
            # For now, we compute loss between first and second half of batch
            batch_size = embeddings.size(0)
            if batch_size >= 2:
                mid = batch_size // 2
                loss = self.loss_fn(embeddings[:mid], embeddings[mid:2*mid])
            else:
                # Skip if batch too small
                continue
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                embeddings = self.model(**batch)
                
                # Same approach as training for consistency
                batch_size = embeddings.size(0)
                if batch_size >= 2:
                    mid = batch_size // 2
                    loss = self.loss_fn(embeddings[:mid], embeddings[mid:2*mid])
                else:
                    continue
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

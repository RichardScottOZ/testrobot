"""
Semantic Search Engine

Provides semantic search functionality using the multimodal model.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss

from .models.multimodal_semantic_model import MultimodalSemanticModel


class SemanticSearchEngine:
    """
    Semantic search engine for multimodal documents.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        model: MultimodalSemanticModel,
        embedding_dim: int = 512,
        index_type: str = 'flat',
        device: str = 'cpu'
    ):
        """
        Initialize semantic search engine.
        
        Args:
            model: Trained multimodal semantic model
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            device: Device to run model on
        """
        self.model = model
        self.embedding_dim = embedding_dim
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize FAISS index
        self.index_type = index_type
        self.index = self._create_index()
        
        # Document metadata storage
        self.documents = []
        self.document_embeddings = []
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on index type."""
        if self.index_type == 'flat':
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)
            
        elif self.index_type == 'ivf':
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            
        elif self.index_type == 'hnsw':
            # Hierarchical navigable small world graph
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> None:
        """
        Add documents to the search index.
        
        Args:
            documents: List of document dictionaries with modality data
            batch_size: Batch size for encoding
        """
        embeddings = []
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare batch data
            batch_data = self._prepare_batch(batch)
            
            # Encode batch
            with torch.no_grad():
                batch_embeddings = self.model(**batch_data)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        # Verify consistency
        if len(all_embeddings) != len(documents):
            raise ValueError(
                f"Embedding count ({len(all_embeddings)}) doesn't match document count ({len(documents)})"
            )
        
        # Add to FAISS index
        if self.index_type == 'ivf' and not self.index.is_trained:
            # Train IVF index if not trained
            self.index.train(all_embeddings)
        
        self.index.add(all_embeddings)
        
        # Store documents and embeddings (maintain consistency)
        self.documents.extend(documents)
        self.document_embeddings.append(all_embeddings)
    
    def search(
        self,
        query: str,
        query_data: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to query.
        
        Args:
            query: Text query string
            query_data: Optional additional query data (preprocessed)
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of matching documents with optional scores
        """
        # Encode query
        with torch.no_grad():
            if query_data is not None:
                # Use provided query data
                query_emb = self.model(**query_data)
            else:
                # Encode text query only
                # Note: In practice, you'd need to tokenize the query first
                raise NotImplementedError("Text tokenization needed - pass query_data instead")
            
            query_vector = query_emb.cpu().numpy()
        
        # Search in FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    'rank': i + 1,
                    'document': self.documents[idx],
                }
                if return_scores:
                    # Convert L2 distance to similarity score
                    result['score'] = float(1.0 / (1.0 + dist))
                    result['distance'] = float(dist)
                results.append(result)
        
        return results
    
    def search_by_document(
        self,
        document_data: Dict[str, Any],
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to a given document.
        
        Args:
            document_data: Document data dictionary
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of similar documents
        """
        # Encode document
        with torch.no_grad():
            batch_data = self._prepare_batch([document_data])
            doc_emb = self.model(**batch_data)
            doc_vector = doc_emb.cpu().numpy()
        
        # Search
        distances, indices = self.index.search(doc_vector, top_k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    'rank': i + 1,
                    'document': self.documents[idx],
                }
                if return_scores:
                    result['score'] = float(1.0 / (1.0 + dist))
                    result['distance'] = float(dist)
                results.append(result)
        
        return results
    
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data for model input.
        
        Args:
            batch: List of document dictionaries
            
        Returns:
            Dictionary of batched tensors
        """
        batch_data = {}
        
        # Helper to stack or pad sequences
        def stack_or_pad(tensors, pad_value=0):
            if not tensors:
                return None
            max_len = max(t.size(0) if len(t.shape) > 0 else 1 for t in tensors)
            padded = []
            for t in tensors:
                if len(t.shape) == 0:
                    t = t.unsqueeze(0)
                if t.size(0) < max_len:
                    padding = torch.full((max_len - t.size(0),) + t.shape[1:], pad_value, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            return torch.stack(padded)
        
        # Process each modality
        for key in ['text_ids', 'text_mask', 'images', 'bboxes', 'bbox_mask',
                    'char_ids', 'char_mask', 'char_styles', 'element_types',
                    'reading_orders', 'hierarchy_levels', 'reading_mask']:
            
            values = [doc.get(key) for doc in batch if key in doc]
            if values:
                if isinstance(values[0], torch.Tensor):
                    batch_data[key] = stack_or_pad(values).to(self.device)
                elif isinstance(values[0], np.ndarray):
                    tensors = [torch.from_numpy(v) for v in values]
                    batch_data[key] = stack_or_pad(tensors).to(self.device)
        
        return batch_data
    
    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        faiss.write_index(self.index, path)
    
    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            'num_documents': len(self.documents),
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.documents)
        }

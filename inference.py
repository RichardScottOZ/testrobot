"""
Inference Script for Multimodal Semantic Search

This script demonstrates how to use the trained model for semantic search.
"""

import torch
import argparse
import json
import os
from PIL import Image

from semantic_search.models import MultimodalSemanticModel
from semantic_search.search_engine import SemanticSearchEngine
from semantic_search.utils import DocumentPreprocessor
from semantic_search.config import get_config


def load_model(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = MultimodalSemanticModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def main(args):
    """Main inference function."""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, device)
    print("Model loaded successfully")
    
    # Initialize search engine
    search_engine = SemanticSearchEngine(
        model=model,
        embedding_dim=config['embedding_dim'],
        index_type=config['search']['index_type'],
        device=device
    )
    
    # Load documents
    if args.documents:
        print(f"Loading documents from {args.documents}...")
        with open(args.documents, 'r') as f:
            documents = json.load(f)
        
        # Initialize preprocessor
        preprocessor = DocumentPreprocessor(config['preprocessing'])
        
        # Preprocess documents
        print(f"Preprocessing {len(documents)} documents...")
        preprocessed_docs = []
        for doc in documents:
            preprocessed = preprocessor.preprocess_document(doc)
            preprocessed_docs.append(preprocessed)
        
        # Add to search engine
        print("Building search index...")
        search_engine.add_documents(preprocessed_docs, batch_size=args.batch_size)
        print(f"Added {len(documents)} documents to search index")
        
        # Save index if requested
        if args.save_index:
            print(f"Saving index to {args.save_index}...")
            search_engine.save_index(args.save_index)
    
    # Load existing index if provided
    if args.load_index:
        print(f"Loading index from {args.load_index}...")
        search_engine.load_index(args.load_index)
    
    # Interactive search mode
    if args.interactive:
        print("\n" + "="*50)
        print("Interactive Search Mode")
        print("="*50)
        print("Enter your search query (or 'quit' to exit)")
        
        preprocessor = DocumentPreprocessor(config['preprocessing'])
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Preprocess query
            query_data = preprocessor.preprocess_document({'text': query})
            
            # Search
            try:
                results = search_engine.search_by_document(
                    query_data,
                    top_k=args.top_k,
                    return_scores=True
                )
                
                # Display results
                print(f"\nTop {len(results)} results:")
                for result in results:
                    print(f"\nRank {result['rank']}:")
                    print(f"  Score: {result['score']:.4f}")
                    print(f"  Distance: {result['distance']:.4f}")
                    if 'text' in result['document']:
                        print(f"  Text: {result['document']['text'][:200]}...")
                    
            except Exception as e:
                print(f"Error during search: {e}")
    
    # Single query mode
    elif args.query:
        print(f"\nSearching for: {args.query}")
        
        preprocessor = DocumentPreprocessor(config['preprocessing'])
        query_data = preprocessor.preprocess_document({'text': args.query})
        
        results = search_engine.search_by_document(
            query_data,
            top_k=args.top_k,
            return_scores=True
        )
        
        # Display results
        print(f"\nTop {len(results)} results:")
        for result in results:
            print(f"\nRank {result['rank']}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Distance: {result['distance']:.4f}")
            if 'text' in result['document']:
                print(f"  Text: {result['document']['text'][:200]}...")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
    
    # Statistics
    stats = search_engine.get_statistics()
    print("\n" + "="*50)
    print("Search Engine Statistics:")
    print(f"  Number of documents: {stats['num_documents']}")
    print(f"  Index type: {stats['index_type']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Search Inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--documents', type=str, default=None,
                        help='Path to documents JSON file')
    parser.add_argument('--load-index', type=str, default=None,
                        help='Path to existing FAISS index')
    parser.add_argument('--save-index', type=str, default=None,
                        help='Path to save FAISS index')
    parser.add_argument('--query', type=str, default=None,
                        help='Search query')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results to return')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for document encoding')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save search results')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    main(args)

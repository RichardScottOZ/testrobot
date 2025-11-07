"""
Example: Basic Usage of Multimodal Semantic Search

This example demonstrates how to use the framework for semantic search.
"""

import torch
from semantic_search import MultimodalSemanticModel, SemanticSearchEngine
from semantic_search.utils import DocumentPreprocessor
from semantic_search.config import get_config


def main():
    """Demonstrate basic usage."""
    
    # 1. Initialize configuration
    print("Initializing configuration...")
    config = get_config()
    
    # 2. Create model
    print("Creating multimodal semantic model...")
    model = MultimodalSemanticModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. Initialize search engine
    print("\nInitializing search engine...")
    search_engine = SemanticSearchEngine(
        model=model,
        embedding_dim=config['embedding_dim'],
        index_type='flat',
        device='cpu'
    )
    
    # 4. Create preprocessor
    preprocessor = DocumentPreprocessor(config['preprocessing'])
    
    # 5. Prepare sample documents
    print("\nPreparing sample documents...")
    documents = [
        {
            'text': 'Deep learning uses neural networks with multiple layers.',
            'bboxes': [[10, 20, 100, 50]],
            'image_width': 800,
            'image_height': 600,
            'ocr_text': 'Deep learning neural networks',
            'elements': [{'type': 'paragraph', 'order': 0, 'level': 0}]
        },
        {
            'text': 'Computer vision enables machines to understand images.',
            'bboxes': [[15, 25, 105, 55]],
            'image_width': 800,
            'image_height': 600,
            'ocr_text': 'Computer vision image understanding',
            'elements': [{'type': 'paragraph', 'order': 0, 'level': 0}]
        },
        {
            'text': 'Natural language processing helps computers understand text.',
            'bboxes': [[12, 22, 102, 52]],
            'image_width': 800,
            'image_height': 600,
            'ocr_text': 'NLP text understanding',
            'elements': [{'type': 'paragraph', 'order': 0, 'level': 0}]
        }
    ]
    
    # 6. Preprocess documents
    print("Preprocessing documents...")
    preprocessed_docs = []
    for doc in documents:
        preprocessed = preprocessor.preprocess_document(doc)
        preprocessed_docs.append(preprocessed)
    
    # 7. Add documents to search index
    print(f"Adding {len(preprocessed_docs)} documents to search index...")
    search_engine.add_documents(preprocessed_docs, batch_size=2)
    
    # 8. Prepare query
    print("\nPreparing search query...")
    query_text = "neural networks and deep learning"
    query_data = preprocessor.preprocess_document({'text': query_text})
    
    # 9. Search
    print(f"Searching for: '{query_text}'")
    results = search_engine.search_by_document(
        query_data,
        top_k=3,
        return_scores=True
    )
    
    # 10. Display results
    print("\n" + "="*50)
    print("Search Results")
    print("="*50)
    for result in results:
        print(f"\nRank {result['rank']}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Text: {result['document']['text']}")
    
    # 11. Get statistics
    stats = search_engine.get_statistics()
    print("\n" + "="*50)
    print("Search Engine Statistics")
    print("="*50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()

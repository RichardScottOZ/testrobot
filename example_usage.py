"""
Simple example demonstrating GPU-accelerated clustering on a raster stack.

This script shows how to use each of the six clustering methods
with a smaller demo raster to verify functionality.
"""

import numpy as np
from gpu_clustering_raster import GPURasterClustering, create_synthetic_raster_stack


def run_simple_example():
    """
    Run a simple example with a small raster stack.
    """
    print("=" * 80)
    print("Simple GPU Clustering Example")
    print("=" * 80)
    
    # Create a small synthetic raster for demonstration
    # Using smaller dimensions for quick testing
    demo_shape = (10, 100, 100)  # 10 bands, 100x100 pixels
    print(f"\nCreating demo raster stack: {demo_shape}")
    raster_stack = create_synthetic_raster_stack(demo_shape)
    
    # Initialize clustering
    clusterer = GPURasterClustering(raster_stack)
    
    print("\n" + "-" * 80)
    print("Testing Method 1: RAPIDS cuML KMeans")
    print("-" * 80)
    try:
        labels_1, model_1 = clusterer.method_1_rapids_kmeans(n_clusters=5)
        if labels_1 is not None:
            print(f"✓ Success! Found {len(np.unique(labels_1))} clusters")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-" * 80)
    print("Testing Method 2: RAPIDS cuML DBSCAN")
    print("-" * 80)
    try:
        labels_2, model_2 = clusterer.method_2_rapids_dbscan(eps=0.3, min_samples=3)
        if labels_2 is not None:
            print(f"✓ Success! Found {len(np.unique(labels_2))} clusters")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-" * 80)
    print("Testing Method 3: RAPIDS cuML HDBSCAN")
    print("-" * 80)
    try:
        labels_3, model_3 = clusterer.method_3_rapids_hdbscan(min_cluster_size=20, min_samples=3)
        if labels_3 is not None:
            print(f"✓ Success! Found {len(np.unique(labels_3))} clusters")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-" * 80)
    print("Testing Method 4: PyTorch KMeans")
    print("-" * 80)
    try:
        labels_4, model_4 = clusterer.method_4_pytorch_kmeans(n_clusters=5)
        if labels_4 is not None:
            print(f"✓ Success! Found {len(np.unique(labels_4))} clusters")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-" * 80)
    print("Testing Method 5: CuPy MiniBatch KMeans")
    print("-" * 80)
    try:
        labels_5, model_5 = clusterer.method_5_cupy_minibatch_kmeans(
            n_clusters=5, batch_size=500
        )
        if labels_5 is not None:
            print(f"✓ Success! Found {len(np.unique(labels_5))} clusters")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "-" * 80)
    print("Testing Method 6: Faiss GPU KMeans")
    print("-" * 80)
    try:
        labels_6, model_6 = clusterer.method_6_faiss_gpu_kmeans(n_clusters=5)
        if labels_6 is not None:
            print(f"✓ Success! Found {len(np.unique(labels_6))} clusters")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("\nNote: Some methods may fail if required libraries are not installed.")
    print("Install missing dependencies from requirements.txt")


def run_full_scale_example():
    """
    Example for full-scale raster (99, 2000, 4000).
    WARNING: Requires significant GPU memory (8GB+)
    """
    print("=" * 80)
    print("Full-Scale Raster Clustering Example")
    print("WARNING: This requires 8GB+ GPU memory")
    print("=" * 80)
    
    # Confirm before proceeding
    response = input("\nDo you want to proceed with full-scale clustering? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Create full-scale raster
    full_shape = (99, 2000, 4000)
    print(f"\nCreating full-scale raster: {full_shape}")
    raster_stack = create_synthetic_raster_stack(full_shape)
    
    # Initialize clustering
    clusterer = GPURasterClustering(raster_stack)
    
    # Use the fastest method (Faiss) for demonstration
    print("\nRunning Faiss GPU KMeans (fastest method)...")
    labels, model = clusterer.method_6_faiss_gpu_kmeans(n_clusters=10)
    
    if labels is not None:
        print(f"\n✓ Success! Clustered {full_shape[1] * full_shape[2]:,} pixels")
        print(f"Found {len(np.unique(labels))} clusters")
        print(f"Output shape: {labels.shape}")
        
        # Save results
        output_file = "cluster_labels.npy"
        np.save(output_file, labels)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    # Run simple example
    run_simple_example()
    
    # Optionally run full-scale example
    print("\n" + "=" * 80)
    run_full_scale_example()

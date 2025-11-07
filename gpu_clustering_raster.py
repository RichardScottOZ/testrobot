"""
GPU-Accelerated Clustering Algorithms for Raster Stacks

This module provides six different GPU-accelerated clustering algorithms
optimized for clustering raster stack data of shape (99, 2000, 4000).

The raster stack has:
- 99 bands (features/channels)
- 2000 rows
- 4000 columns
- Total pixels: 8,000,000

Each algorithm is designed to handle the large-scale data efficiently using GPU acceleration.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class GPURasterClustering:
    """
    A collection of GPU-accelerated clustering algorithms for raster data.
    
    Each method handles reshaping the raster stack from (bands, rows, cols)
    to (n_samples, n_features) format required by clustering algorithms.
    """
    
    def __init__(self, raster_stack: np.ndarray):
        """
        Initialize with a raster stack.
        
        Parameters
        ----------
        raster_stack : np.ndarray
            Shape (n_bands, n_rows, n_cols), e.g., (99, 2000, 4000)
        """
        self.raster_stack = raster_stack
        self.n_bands, self.n_rows, self.n_cols = raster_stack.shape
        self.n_pixels = self.n_rows * self.n_cols
        
        # Reshape to (n_samples, n_features)
        self.data_matrix = raster_stack.reshape(self.n_bands, -1).T
        print(f"Raster shape: {raster_stack.shape}")
        print(f"Data matrix shape: {self.data_matrix.shape}")
    
    def reshape_labels_to_raster(self, labels: np.ndarray) -> np.ndarray:
        """
        Reshape 1D labels back to raster format.
        
        Parameters
        ----------
        labels : np.ndarray
            1D array of cluster labels
            
        Returns
        -------
        np.ndarray
            Labels reshaped to (n_rows, n_cols)
        """
        return labels.reshape(self.n_rows, self.n_cols)
    
    def method_1_rapids_kmeans(self, n_clusters: int = 10, max_iter: int = 300) -> Tuple[np.ndarray, object]:
        """
        Method 1: RAPIDS cuML KMeans
        
        Uses NVIDIA RAPIDS cuML for GPU-accelerated KMeans clustering.
        Very fast for large datasets with excellent GPU utilization.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels reshaped to (n_rows, n_cols)
        model : object
            Fitted clustering model
        """
        try:
            from cuml.cluster import KMeans as cuKMeans
            import cudf
            
            print("\n=== Method 1: RAPIDS cuML KMeans ===")
            
            # Convert to cuDF DataFrame for better GPU memory management
            model = cuKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
            
            # Fit and predict
            labels = model.fit_predict(self.data_matrix)
            
            # Convert back to numpy if needed
            if hasattr(labels, 'to_numpy'):
                labels = labels.to_numpy()
            
            labels_raster = self.reshape_labels_to_raster(labels)
            
            print(f"Clusters: {n_clusters}")
            print(f"Unique labels: {len(np.unique(labels_raster))}")
            print(f"Output shape: {labels_raster.shape}")
            
            return labels_raster, model
            
        except ImportError as e:
            print(f"Error: RAPIDS cuML not available. Install with: pip install cuml")
            print(f"Details: {e}")
            return None, None
    
    def method_2_rapids_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, object]:
        """
        Method 2: RAPIDS cuML DBSCAN
        
        Density-based clustering using GPU acceleration.
        Good for finding clusters of arbitrary shape and identifying outliers.
        
        Parameters
        ----------
        eps : float
            Maximum distance between samples
        min_samples : int
            Minimum samples in a neighborhood
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels reshaped to (n_rows, n_cols)
        model : object
            Fitted clustering model
        """
        try:
            from cuml.cluster import DBSCAN as cuDBSCAN
            
            print("\n=== Method 2: RAPIDS cuML DBSCAN ===")
            
            model = cuDBSCAN(eps=eps, min_samples=min_samples)
            
            # Fit and predict
            labels = model.fit_predict(self.data_matrix)
            
            # Convert back to numpy if needed
            if hasattr(labels, 'to_numpy'):
                labels = labels.to_numpy()
            
            labels_raster = self.reshape_labels_to_raster(labels)
            
            print(f"eps: {eps}, min_samples: {min_samples}")
            print(f"Unique labels: {len(np.unique(labels_raster))}")
            print(f"Noise points (label=-1): {np.sum(labels_raster == -1)}")
            print(f"Output shape: {labels_raster.shape}")
            
            return labels_raster, model
            
        except ImportError as e:
            print(f"Error: RAPIDS cuML not available. Install with: pip install cuml")
            print(f"Details: {e}")
            return None, None
    
    def method_3_rapids_hdbscan(self, min_cluster_size: int = 100, min_samples: int = 5) -> Tuple[np.ndarray, object]:
        """
        Method 3: RAPIDS cuML HDBSCAN
        
        Hierarchical DBSCAN with automatic cluster number detection.
        Excellent for complex spatial patterns in raster data.
        
        Parameters
        ----------
        min_cluster_size : int
            Minimum size of clusters
        min_samples : int
            Minimum samples in a neighborhood
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels reshaped to (n_rows, n_cols)
        model : object
            Fitted clustering model
        """
        try:
            from cuml.cluster import HDBSCAN as cuHDBSCAN
            
            print("\n=== Method 3: RAPIDS cuML HDBSCAN ===")
            
            model = cuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                gen_min_span_tree=True
            )
            
            # Fit and predict
            labels = model.fit_predict(self.data_matrix)
            
            # Convert back to numpy if needed
            if hasattr(labels, 'to_numpy'):
                labels = labels.to_numpy()
            
            labels_raster = self.reshape_labels_to_raster(labels)
            
            print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
            print(f"Unique labels: {len(np.unique(labels_raster))}")
            print(f"Noise points (label=-1): {np.sum(labels_raster == -1)}")
            print(f"Output shape: {labels_raster.shape}")
            
            return labels_raster, model
            
        except ImportError as e:
            print(f"Error: RAPIDS cuML not available. Install with: pip install cuml")
            print(f"Details: {e}")
            return None, None
    
    def method_4_pytorch_kmeans(self, n_clusters: int = 10, max_iter: int = 300) -> Tuple[np.ndarray, object]:
        """
        Method 4: PyTorch-based KMeans
        
        Custom KMeans implementation using PyTorch for GPU acceleration.
        Flexible and integrates well with deep learning workflows.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels reshaped to (n_rows, n_cols)
        model : dict
            Dictionary containing centroids and other model info
        """
        try:
            import torch
            
            print("\n=== Method 4: PyTorch KMeans ===")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Convert data to torch tensor
            X = torch.from_numpy(self.data_matrix.astype(np.float32)).to(device)
            
            # Initialize centroids using kmeans++
            n_samples = X.shape[0]
            indices = torch.randperm(n_samples)[:n_clusters]
            centroids = X[indices].clone()
            
            iteration = 0
            for iteration in range(max_iter):
                # Compute distances to centroids
                distances = torch.cdist(X, centroids)
                
                # Assign to nearest centroid
                labels = torch.argmin(distances, dim=1)
                
                # Update centroids
                old_centroids = centroids.clone()
                for k in range(n_clusters):
                    mask = labels == k
                    if mask.sum() > 0:
                        centroids[k] = X[mask].mean(dim=0)
                
                # Check convergence
                if torch.allclose(centroids, old_centroids, rtol=1e-4):
                    print(f"Converged at iteration {iteration}")
                    break
            
            # Convert labels back to numpy
            labels_np = labels.cpu().numpy()
            labels_raster = self.reshape_labels_to_raster(labels_np)
            
            model = {
                'centroids': centroids.cpu().numpy(),
                'n_clusters': n_clusters,
                'iterations': iteration + 1
            }
            
            print(f"Clusters: {n_clusters}")
            print(f"Unique labels: {len(np.unique(labels_raster))}")
            print(f"Output shape: {labels_raster.shape}")
            
            return labels_raster, model
            
        except ImportError as e:
            print(f"Error: PyTorch not available. Install with: pip install torch")
            print(f"Details: {e}")
            return None, None
    
    def method_5_cupy_minibatch_kmeans(self, n_clusters: int = 10, batch_size: int = 10000, max_iter: int = 100) -> Tuple[np.ndarray, object]:
        """
        Method 5: CuPy-based MiniBatch KMeans
        
        MiniBatch KMeans using CuPy for memory-efficient GPU clustering.
        Ideal for very large datasets that don't fit in GPU memory.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        batch_size : int
            Size of mini-batches
        max_iter : int
            Maximum number of iterations
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels reshaped to (n_rows, n_cols)
        model : dict
            Dictionary containing centroids and other model info
        """
        try:
            import cupy as cp
            
            print("\n=== Method 5: CuPy MiniBatch KMeans ===")
            
            # Convert data to CuPy array
            X = cp.asarray(self.data_matrix.astype(np.float32))
            n_samples = X.shape[0]
            
            # Initialize centroids
            indices = cp.random.permutation(n_samples)[:n_clusters]
            centroids = X[indices].copy()
            
            # MiniBatch KMeans
            for iteration in range(max_iter):
                # Random mini-batch
                batch_indices = cp.random.permutation(n_samples)[:batch_size]
                X_batch = X[batch_indices]
                
                # Assign to nearest centroid
                distances = cp.linalg.norm(X_batch[:, cp.newaxis] - centroids, axis=2)
                labels_batch = cp.argmin(distances, axis=1)
                
                # Update centroids using mini-batch
                for k in range(n_clusters):
                    mask = labels_batch == k
                    if cp.sum(mask) > 0:
                        centroids[k] = 0.9 * centroids[k] + 0.1 * X_batch[mask].mean(axis=0)
            
            # Final assignment for all data
            distances = cp.linalg.norm(X[:, cp.newaxis] - centroids, axis=2)
            labels = cp.argmin(distances, axis=1)
            
            # Convert back to numpy
            labels_np = cp.asnumpy(labels)
            labels_raster = self.reshape_labels_to_raster(labels_np)
            
            model = {
                'centroids': cp.asnumpy(centroids),
                'n_clusters': n_clusters,
                'batch_size': batch_size,
                'iterations': max_iter
            }
            
            print(f"Clusters: {n_clusters}, Batch size: {batch_size}")
            print(f"Unique labels: {len(np.unique(labels_raster))}")
            print(f"Output shape: {labels_raster.shape}")
            
            return labels_raster, model
            
        except ImportError as e:
            print(f"Error: CuPy not available. Install with: pip install cupy-cuda11x (for CUDA 11.x) or cupy-cuda12x (for CUDA 12.x)")
            print(f"Details: {e}")
            return None, None
    
    def method_6_faiss_gpu_kmeans(self, n_clusters: int = 10, max_iter: int = 300, nredo: int = 1) -> Tuple[np.ndarray, object]:
        """
        Method 6: Faiss GPU KMeans
        
        Facebook's Faiss library optimized for similarity search and clustering.
        Extremely fast and memory-efficient for large-scale clustering.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
        nredo : int
            Number of times to redo clustering (best result is kept)
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels reshaped to (n_rows, n_cols)
        model : object
            Faiss clustering object
        """
        try:
            import faiss
            
            print("\n=== Method 6: Faiss GPU KMeans ===")
            
            # Prepare data (Faiss requires C-contiguous float32)
            X = np.ascontiguousarray(self.data_matrix.astype(np.float32))
            n_samples, n_features = X.shape
            
            # Initialize Faiss clustering
            kmeans = faiss.Clustering(n_features, n_clusters)
            kmeans.niter = max_iter
            kmeans.nredo = nredo
            kmeans.verbose = True
            
            # Use GPU
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(n_features)
            
            # Use GPU index
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # Train
            kmeans.train(X, gpu_index)
            
            # Get centroids
            centroids = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, n_features)
            
            # Assign labels
            gpu_index.reset()
            gpu_index.add(centroids)
            _, labels = gpu_index.search(X, 1)
            labels = labels.flatten()
            
            labels_raster = self.reshape_labels_to_raster(labels)
            
            print(f"Clusters: {n_clusters}")
            print(f"Unique labels: {len(np.unique(labels_raster))}")
            print(f"Output shape: {labels_raster.shape}")
            
            return labels_raster, kmeans
            
        except ImportError as e:
            print(f"Error: Faiss not available. Install with: pip install faiss-gpu")
            print(f"Details: {e}")
            return None, None


def create_synthetic_raster_stack(shape: Tuple[int, int, int] = (99, 2000, 4000)) -> np.ndarray:
    """
    Create a synthetic raster stack for testing.
    
    Parameters
    ----------
    shape : tuple
        Shape of raster stack (n_bands, n_rows, n_cols)
        
    Returns
    -------
    np.ndarray
        Synthetic raster stack
    """
    n_bands, n_rows, n_cols = shape
    print(f"Creating synthetic raster stack of shape {shape}")
    print(f"Total size: {n_bands * n_rows * n_cols * 4 / 1e9:.2f} GB (float32)")
    
    # Create synthetic data with some spatial structure
    # Use smaller intermediate arrays to save memory
    raster = np.zeros(shape, dtype=np.float32)
    
    for i in range(n_bands):
        # Create patterns that vary by band
        x = np.linspace(0, 4 * np.pi, n_cols)
        y = np.linspace(0, 4 * np.pi, n_rows)
        xx, yy = np.meshgrid(x, y)
        
        # Combination of sine waves with different frequencies
        raster[i] = (
            np.sin(xx / (i + 1)) * np.cos(yy / (i + 1)) +
            np.random.randn(n_rows, n_cols) * 0.1
        )
    
    return raster


def main():
    """
    Main function demonstrating all six GPU-accelerated clustering methods.
    """
    print("=" * 80)
    print("GPU-Accelerated Clustering for Raster Stacks")
    print("=" * 80)
    
    # Create synthetic raster stack
    # For demonstration, use smaller size to avoid memory issues
    # Full size would be (99, 2000, 4000)
    demo_shape = (99, 200, 400)  # Smaller for demo
    raster_stack = create_synthetic_raster_stack(demo_shape)
    
    # Initialize clustering object
    clusterer = GPURasterClustering(raster_stack)
    
    # Method 1: RAPIDS cuML KMeans
    labels_1, model_1 = clusterer.method_1_rapids_kmeans(n_clusters=5)
    
    # Method 2: RAPIDS cuML DBSCAN
    labels_2, model_2 = clusterer.method_2_rapids_dbscan(eps=0.5, min_samples=5)
    
    # Method 3: RAPIDS cuML HDBSCAN
    labels_3, model_3 = clusterer.method_3_rapids_hdbscan(min_cluster_size=50, min_samples=5)
    
    # Method 4: PyTorch KMeans
    labels_4, model_4 = clusterer.method_4_pytorch_kmeans(n_clusters=5)
    
    # Method 5: CuPy MiniBatch KMeans
    labels_5, model_5 = clusterer.method_5_cupy_minibatch_kmeans(n_clusters=5, batch_size=1000)
    
    # Method 6: Faiss GPU KMeans
    labels_6, model_6 = clusterer.method_6_faiss_gpu_kmeans(n_clusters=5)
    
    print("\n" + "=" * 80)
    print("All clustering methods completed!")
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    print(f"Input raster shape: {raster_stack.shape}")
    print(f"Output labels shape for each method: ({demo_shape[1]}, {demo_shape[2]})")
    print("\nNote: For full-scale raster (99, 2000, 4000), use production GPU with sufficient memory.")
    print("Recommended: NVIDIA A100 (40GB+) or V100 (32GB+)")


if __name__ == "__main__":
    main()

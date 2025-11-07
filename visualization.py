"""
Visualization utilities for clustering results.

This module provides functions to visualize and compare clustering results
from different GPU-accelerated methods.
"""

import numpy as np
from typing import List, Tuple, Optional


def visualize_clusters(labels: np.ndarray, title: str = "Cluster Map", 
                       save_path: Optional[str] = None):
    """
    Visualize cluster labels as a 2D map.
    
    Parameters
    ----------
    labels : np.ndarray
        2D array of cluster labels
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get unique labels
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Create colormap
        if n_clusters <= 10:
            cmap = plt.cm.tab10
        elif n_clusters <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral
        
        # Plot
        im = plt.imshow(labels, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, label='Cluster ID')
        plt.title(f"{title}\n({n_clusters} clusters)")
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")


def compare_clustering_results(results_dict: dict, save_path: Optional[str] = None):
    """
    Compare multiple clustering results side by side.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and label arrays as values
    save_path : str, optional
        Path to save the comparison figure
    """
    try:
        import matplotlib.pyplot as plt
        
        n_methods = len(results_dict)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (method_name, labels) in enumerate(results_dict.items()):
            if idx >= 6:
                break
                
            if labels is None:
                axes[idx].text(0.5, 0.5, f"{method_name}\n(Failed)", 
                             ha='center', va='center', fontsize=12)
                axes[idx].set_title(method_name)
                continue
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Choose colormap
            if n_clusters <= 10:
                cmap = plt.cm.tab10
            elif n_clusters <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.nipy_spectral
            
            im = axes[idx].imshow(labels, cmap=cmap, interpolation='nearest')
            axes[idx].set_title(f"{method_name}\n({n_clusters} clusters)")
            axes[idx].set_xlabel('Column')
            axes[idx].set_ylabel('Row')
            plt.colorbar(im, ax=axes[idx], label='Cluster ID')
        
        # Hide unused subplots
        for idx in range(n_methods, 6):
            axes[idx].axis('off')
        
        plt.suptitle('Clustering Methods Comparison', fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")


def plot_cluster_statistics(labels: np.ndarray, method_name: str = "Method",
                           save_path: Optional[str] = None):
    """
    Plot statistics about cluster sizes and distribution.
    
    Parameters
    ----------
    labels : np.ndarray
        2D array of cluster labels
    method_name : str
        Name of the clustering method
    save_path : str, optional
        Path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        
        # Calculate statistics
        unique_labels = np.unique(labels)
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot of cluster sizes
        ax1.bar(range(len(cluster_sizes)), cluster_sizes)
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Pixels')
        ax1.set_title(f'Cluster Sizes - {method_name}')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of cluster sizes
        ax2.hist(cluster_sizes, bins=20, edgecolor='black')
        ax2.set_xlabel('Cluster Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Cluster Sizes')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total clusters: {len(unique_labels)}\n"
        stats_text += f"Min size: {min(cluster_sizes):,}\n"
        stats_text += f"Max size: {max(cluster_sizes):,}\n"
        stats_text += f"Mean size: {np.mean(cluster_sizes):,.0f}\n"
        stats_text += f"Median size: {np.median(cluster_sizes):,.0f}"
        
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved statistics plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")


def create_rgb_composite_with_clusters(raster_stack: np.ndarray, 
                                       labels: np.ndarray,
                                       rgb_bands: Tuple[int, int, int] = (0, 1, 2),
                                       save_path: Optional[str] = None):
    """
    Create RGB composite with cluster boundaries overlay.
    
    Parameters
    ----------
    raster_stack : np.ndarray
        Original raster stack (bands, rows, cols)
    labels : np.ndarray
        Cluster labels (rows, cols)
    rgb_bands : tuple
        Indices of bands to use for RGB (R, G, B)
    save_path : str, optional
        Path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.ndimage import generic_gradient_magnitude
        
        # Extract RGB bands
        r_band = raster_stack[rgb_bands[0]]
        g_band = raster_stack[rgb_bands[1]]
        b_band = raster_stack[rgb_bands[2]]
        
        # Normalize to 0-1 for display
        def normalize(band):
            min_val, max_val = np.percentile(band, [2, 98])
            return np.clip((band - min_val) / (max_val - min_val), 0, 1)
        
        rgb = np.dstack([normalize(r_band), normalize(g_band), normalize(b_band)])
        
        # Find cluster boundaries
        boundaries = generic_gradient_magnitude(labels.astype(float), 
                                               np.ones((3, 3)))
        boundaries = boundaries > 0
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # RGB composite
        ax1.imshow(rgb)
        ax1.set_title('RGB Composite')
        ax1.axis('off')
        
        # Cluster map
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        cmap = plt.cm.tab20 if n_clusters <= 20 else plt.cm.nipy_spectral
        
        im = ax2.imshow(labels, cmap=cmap, interpolation='nearest')
        ax2.set_title(f'Clusters ({n_clusters})')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Cluster ID', fraction=0.046)
        
        # RGB with boundaries
        rgb_with_boundaries = rgb.copy()
        rgb_with_boundaries[boundaries] = [1, 0, 0]  # Red boundaries
        
        ax3.imshow(rgb_with_boundaries)
        ax3.set_title('RGB with Cluster Boundaries')
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved RGB composite to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError as e:
        print(f"Required libraries not installed: {e}")
        print("Install with: pip install matplotlib scipy")


def main():
    """
    Demonstration of visualization functions.
    """
    from gpu_clustering_raster import GPURasterClustering, create_synthetic_raster_stack
    
    print("Creating demo data...")
    raster_stack = create_synthetic_raster_stack((10, 200, 300))
    
    clusterer = GPURasterClustering(raster_stack)
    
    # Try PyTorch method (most likely to work without special GPU libraries)
    print("\nRunning clustering...")
    try:
        import torch
        labels, model = clusterer.method_4_pytorch_kmeans(n_clusters=7)
        
        if labels is not None:
            print("\nGenerating visualizations...")
            
            # Single cluster map
            visualize_clusters(labels, "PyTorch KMeans Clustering", 
                             "cluster_map.png")
            
            # Cluster statistics
            plot_cluster_statistics(labels, "PyTorch KMeans", 
                                  "cluster_stats.png")
            
            # RGB composite with clusters
            create_rgb_composite_with_clusters(raster_stack, labels, 
                                             rgb_bands=(0, 3, 6),
                                             save_path="rgb_composite.png")
            
            print("\nVisualization complete!")
        else:
            print("Clustering failed")
            
    except ImportError:
        print("\nPyTorch not available. Install with: pip install torch")
        print("Creating demo visualization with random data...")
        
        # Create random cluster labels for demo
        labels = np.random.randint(0, 5, size=(200, 300))
        visualize_clusters(labels, "Demo Cluster Map", "demo_clusters.png")


if __name__ == "__main__":
    main()

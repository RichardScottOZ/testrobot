"""
Comparison of GPU-accelerated clustering methods.

This script compares performance, memory usage, and results
of all six clustering methods on the same dataset.
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from gpu_clustering_raster import GPURasterClustering, create_synthetic_raster_stack


class ClusteringBenchmark:
    """Benchmark different GPU clustering methods."""
    
    def __init__(self, raster_stack: np.ndarray):
        self.raster_stack = raster_stack
        self.clusterer = GPURasterClustering(raster_stack)
        self.results = {}
    
    def benchmark_method(self, method_name: str, method_func, **kwargs) -> Dict:
        """
        Benchmark a single clustering method.
        
        Parameters
        ----------
        method_name : str
            Name of the method
        method_func : callable
            Method to benchmark
        **kwargs : dict
            Arguments for the method
            
        Returns
        -------
        dict
            Benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*60}")
        
        try:
            # Time the execution
            start_time = time.time()
            labels, model = method_func(**kwargs)
            end_time = time.time()
            
            if labels is None:
                return {
                    'method': method_name,
                    'success': False,
                    'error': 'Method returned None'
                }
            
            elapsed_time = end_time - start_time
            
            # Calculate statistics
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Count pixels per cluster
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            
            result = {
                'method': method_name,
                'success': True,
                'time_seconds': elapsed_time,
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes,
                'min_cluster_size': min(cluster_sizes),
                'max_cluster_size': max(cluster_sizes),
                'mean_cluster_size': np.mean(cluster_sizes),
                'labels': labels,
                'model': model
            }
            
            print(f"✓ Completed in {elapsed_time:.2f} seconds")
            print(f"  Clusters found: {n_clusters}")
            print(f"  Cluster size range: {min(cluster_sizes)} - {max(cluster_sizes)}")
            
            return result
            
        except Exception as e:
            print(f"✗ Failed with error: {e}")
            return {
                'method': method_name,
                'success': False,
                'error': str(e)
            }
    
    def run_all_benchmarks(self, n_clusters: int = 5) -> Dict[str, Dict]:
        """
        Run benchmarks for all methods.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters for methods that require it
            
        Returns
        -------
        dict
            Dictionary of results for each method
        """
        results = {}
        
        # Method 1: RAPIDS KMeans
        results['rapids_kmeans'] = self.benchmark_method(
            'RAPIDS cuML KMeans',
            self.clusterer.method_1_rapids_kmeans,
            n_clusters=n_clusters
        )
        
        # Method 2: RAPIDS DBSCAN
        results['rapids_dbscan'] = self.benchmark_method(
            'RAPIDS cuML DBSCAN',
            self.clusterer.method_2_rapids_dbscan,
            eps=0.5,
            min_samples=5
        )
        
        # Method 3: RAPIDS HDBSCAN
        results['rapids_hdbscan'] = self.benchmark_method(
            'RAPIDS cuML HDBSCAN',
            self.clusterer.method_3_rapids_hdbscan,
            min_cluster_size=50,
            min_samples=5
        )
        
        # Method 4: PyTorch KMeans
        results['pytorch_kmeans'] = self.benchmark_method(
            'PyTorch KMeans',
            self.clusterer.method_4_pytorch_kmeans,
            n_clusters=n_clusters
        )
        
        # Method 5: CuPy MiniBatch KMeans
        results['cupy_minibatch'] = self.benchmark_method(
            'CuPy MiniBatch KMeans',
            self.clusterer.method_5_cupy_minibatch_kmeans,
            n_clusters=n_clusters,
            batch_size=1000
        )
        
        # Method 6: Faiss GPU KMeans
        results['faiss_kmeans'] = self.benchmark_method(
            'Faiss GPU KMeans',
            self.clusterer.method_6_faiss_gpu_kmeans,
            n_clusters=n_clusters
        )
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Filter successful results
        successful = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if not successful:
            print("No methods completed successfully.")
            return
        
        # Sort by execution time
        sorted_results = sorted(
            successful.items(),
            key=lambda x: x[1]['time_seconds']
        )
        
        print(f"\n{'Method':<30} {'Time (s)':<12} {'Clusters':<10} {'Status'}")
        print("-" * 80)
        
        for method_key, result in sorted_results:
            print(f"{result['method']:<30} {result['time_seconds']:<12.2f} "
                  f"{result['n_clusters']:<10} {'✓ Success'}")
        
        # Print failed methods
        failed = {k: v for k, v in self.results.items() if not v.get('success', False)}
        if failed:
            print("\nFailed Methods:")
            print("-" * 80)
            for method_key, result in failed.items():
                print(f"{result['method']:<30} ✗ {result.get('error', 'Unknown error')}")
        
        # Performance metrics
        if len(successful) > 1:
            times = [r['time_seconds'] for r in successful.values()]
            fastest = min(times)
            slowest = max(times)
            
            print(f"\nPerformance Range:")
            print(f"  Fastest: {fastest:.2f}s")
            print(f"  Slowest: {slowest:.2f}s")
            print(f"  Speedup: {slowest/fastest:.2f}x")


def main():
    """Main comparison function."""
    print("=" * 80)
    print("GPU Clustering Methods Comparison")
    print("=" * 80)
    
    # Create demo raster
    demo_shape = (20, 200, 200)  # 20 bands, 200x200 pixels
    print(f"\nCreating demo raster: {demo_shape}")
    raster_stack = create_synthetic_raster_stack(demo_shape)
    
    # Run benchmarks
    benchmark = ClusteringBenchmark(raster_stack)
    results = benchmark.run_all_benchmarks(n_clusters=5)
    
    # Print summary
    benchmark.print_summary()
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("""
1. For fastest KMeans clustering: Use Faiss GPU or RAPIDS cuML KMeans
2. For density-based clustering: Use RAPIDS DBSCAN or HDBSCAN
3. For memory efficiency: Use CuPy MiniBatch KMeans
4. For PyTorch integration: Use PyTorch KMeans
5. For production systems: Use RAPIDS cuML (well-supported, stable)
6. For research/flexibility: Use PyTorch or CuPy implementations
    """)


if __name__ == "__main__":
    main()

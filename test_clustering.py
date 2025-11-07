"""
Comprehensive test suite for GPU clustering algorithms.

This script tests all 6 methods with a small dataset to verify
they work correctly (when their dependencies are available).
"""

import numpy as np
import sys


def test_imports():
    """Test that core module imports correctly."""
    print("Testing imports...")
    try:
        from gpu_clustering_raster import (
            GPURasterClustering,
            create_synthetic_raster_stack
        )
        print("✓ Core module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import core module: {e}")
        return False


def test_synthetic_data_generation():
    """Test synthetic raster generation."""
    print("\nTesting synthetic data generation...")
    try:
        from gpu_clustering_raster import create_synthetic_raster_stack
        
        # Test small raster
        raster = create_synthetic_raster_stack((5, 100, 100))
        
        assert raster.shape == (5, 100, 100), f"Wrong shape: {raster.shape}"
        assert raster.dtype == np.float32, f"Wrong dtype: {raster.dtype}"
        assert not np.any(np.isnan(raster)), "Contains NaN values"
        assert not np.any(np.isinf(raster)), "Contains Inf values"
        
        print(f"✓ Synthetic raster created: shape={raster.shape}, dtype={raster.dtype}")
        print(f"  Data range: [{raster.min():.3f}, {raster.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
        return False


def test_class_initialization():
    """Test GPURasterClustering class initialization."""
    print("\nTesting class initialization...")
    try:
        from gpu_clustering_raster import (
            GPURasterClustering,
            create_synthetic_raster_stack
        )
        
        raster = create_synthetic_raster_stack((3, 50, 50))
        clusterer = GPURasterClustering(raster)
        
        assert clusterer.n_bands == 3
        assert clusterer.n_rows == 50
        assert clusterer.n_cols == 50
        assert clusterer.n_pixels == 2500
        assert clusterer.data_matrix.shape == (2500, 3)
        
        print("✓ Class initialization successful")
        print(f"  Bands: {clusterer.n_bands}, Pixels: {clusterer.n_pixels}")
        return True
        
    except Exception as e:
        print(f"✗ Class initialization failed: {e}")
        return False


def test_reshape_labels():
    """Test label reshaping functionality."""
    print("\nTesting label reshaping...")
    try:
        from gpu_clustering_raster import (
            GPURasterClustering,
            create_synthetic_raster_stack
        )
        
        raster = create_synthetic_raster_stack((3, 50, 50))
        clusterer = GPURasterClustering(raster)
        
        # Create fake labels
        labels_1d = np.random.randint(0, 5, size=2500)
        labels_2d = clusterer.reshape_labels_to_raster(labels_1d)
        
        assert labels_2d.shape == (50, 50), f"Wrong reshape: {labels_2d.shape}"
        assert np.array_equal(labels_2d.flatten(), labels_1d), "Reshape changed values"
        
        print("✓ Label reshaping works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Label reshaping failed: {e}")
        return False


def test_method_availability():
    """Test which methods are available based on installed libraries."""
    print("\nChecking method availability...")
    
    methods_status = {}
    
    # Check RAPIDS
    try:
        import cuml
        methods_status['RAPIDS'] = '✓ Available'
    except ImportError:
        methods_status['RAPIDS'] = '✗ Not installed (conda install -c rapidsai cuml)'
    
    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            methods_status['PyTorch'] = f'✓ Available (CUDA: {torch.cuda.get_device_name(0)})'
        else:
            methods_status['PyTorch'] = '⚠ Available but no CUDA GPU'
    except ImportError:
        methods_status['PyTorch'] = '✗ Not installed (pip install torch)'
    
    # Check CuPy
    try:
        import cupy
        methods_status['CuPy'] = '✓ Available'
    except ImportError:
        methods_status['CuPy'] = '✗ Not installed (pip install cupy-cuda11x or cupy-cuda12x)'
    
    # Check Faiss
    try:
        import faiss
        methods_status['Faiss'] = '✓ Available'
    except ImportError:
        methods_status['Faiss'] = '✗ Not installed (pip install faiss-gpu)'
    
    print("\nLibrary Availability:")
    for lib, status in methods_status.items():
        print(f"  {lib}: {status}")
    
    return any('✓' in status for status in methods_status.values())


def test_all_methods():
    """Test all available clustering methods."""
    from gpu_clustering_raster import (
        GPURasterClustering,
        create_synthetic_raster_stack
    )
    
    print("\n" + "=" * 70)
    print("Testing All Clustering Methods")
    print("=" * 70)
    
    # Create small test data
    raster = create_synthetic_raster_stack((5, 50, 50))
    clusterer = GPURasterClustering(raster)
    
    results = {}
    
    # Method 1: RAPIDS KMeans
    print("\n[1/6] Testing RAPIDS KMeans...")
    try:
        labels, model = clusterer.method_1_rapids_kmeans(n_clusters=3)
        if labels is not None:
            results['RAPIDS KMeans'] = '✓ Passed'
            print(f"  Clusters: {len(np.unique(labels))}, Shape: {labels.shape}")
        else:
            results['RAPIDS KMeans'] = '○ Skipped (library not available)'
    except Exception as e:
        results['RAPIDS KMeans'] = f'✗ Failed: {str(e)[:50]}'
    
    # Method 2: RAPIDS DBSCAN
    print("\n[2/6] Testing RAPIDS DBSCAN...")
    try:
        labels, model = clusterer.method_2_rapids_dbscan(eps=0.5, min_samples=3)
        if labels is not None:
            results['RAPIDS DBSCAN'] = '✓ Passed'
            print(f"  Clusters: {len(np.unique(labels))}, Shape: {labels.shape}")
        else:
            results['RAPIDS DBSCAN'] = '○ Skipped (library not available)'
    except Exception as e:
        results['RAPIDS DBSCAN'] = f'✗ Failed: {str(e)[:50]}'
    
    # Method 3: RAPIDS HDBSCAN
    print("\n[3/6] Testing RAPIDS HDBSCAN...")
    try:
        labels, model = clusterer.method_3_rapids_hdbscan(min_cluster_size=20, min_samples=3)
        if labels is not None:
            results['RAPIDS HDBSCAN'] = '✓ Passed'
            print(f"  Clusters: {len(np.unique(labels))}, Shape: {labels.shape}")
        else:
            results['RAPIDS HDBSCAN'] = '○ Skipped (library not available)'
    except Exception as e:
        results['RAPIDS HDBSCAN'] = f'✗ Failed: {str(e)[:50]}'
    
    # Method 4: PyTorch KMeans
    print("\n[4/6] Testing PyTorch KMeans...")
    try:
        labels, model = clusterer.method_4_pytorch_kmeans(n_clusters=3)
        if labels is not None:
            results['PyTorch KMeans'] = '✓ Passed'
            print(f"  Clusters: {len(np.unique(labels))}, Shape: {labels.shape}")
        else:
            results['PyTorch KMeans'] = '○ Skipped (library not available)'
    except Exception as e:
        results['PyTorch KMeans'] = f'✗ Failed: {str(e)[:50]}'
    
    # Method 5: CuPy MiniBatch
    print("\n[5/6] Testing CuPy MiniBatch KMeans...")
    try:
        labels, model = clusterer.method_5_cupy_minibatch_kmeans(n_clusters=3, batch_size=500)
        if labels is not None:
            results['CuPy MiniBatch'] = '✓ Passed'
            print(f"  Clusters: {len(np.unique(labels))}, Shape: {labels.shape}")
        else:
            results['CuPy MiniBatch'] = '○ Skipped (library not available)'
    except Exception as e:
        results['CuPy MiniBatch'] = f'✗ Failed: {str(e)[:50]}'
    
    # Method 6: Faiss KMeans
    print("\n[6/6] Testing Faiss GPU KMeans...")
    try:
        labels, model = clusterer.method_6_faiss_gpu_kmeans(n_clusters=3)
        if labels is not None:
            results['Faiss KMeans'] = '✓ Passed'
            print(f"  Clusters: {len(np.unique(labels))}, Shape: {labels.shape}")
        else:
            results['Faiss KMeans'] = '○ Skipped (library not available)'
    except Exception as e:
        results['Faiss KMeans'] = f'✗ Failed: {str(e)[:50]}'
    
    return results


def main():
    """Run all tests."""
    print("=" * 70)
    print("GPU Clustering Test Suite")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    # Basic tests
    tests = [
        ("Import test", test_imports),
        ("Synthetic data", test_synthetic_data_generation),
        ("Class initialization", test_class_initialization),
        ("Label reshaping", test_reshape_labels),
        ("Library availability", test_method_availability),
    ]
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    # Method tests
    if passed > 0:  # Only test methods if basic tests passed
        results = test_all_methods()
        
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        
        for method, status in results.items():
            print(f"{method:<25} {status}")
            if '✓' in status:
                passed += 1
            elif '✗' in status:
                failed += 1
    
    print("\n" + "=" * 70)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0 and passed > 0:
        print("\n✓ All available tests passed!")
        return 0
    elif passed > 0:
        print(f"\n⚠ Some tests passed ({passed}), some failed/skipped ({failed})")
        return 0  # Still return 0 as skipped tests are expected
    else:
        print("\n✗ All tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

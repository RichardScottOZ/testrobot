# Quick Start Guide

Get started with GPU-accelerated clustering in 5 minutes!

## Installation

### Step 1: Install Basic Dependencies

```bash
pip install numpy torch
```

### Step 2: Choose Your GPU Library

**Option A: PyTorch (Easiest)**
```bash
# Already installed above
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Option B: Faiss (Fastest)**
```bash
pip install faiss-gpu
```

**Option C: RAPIDS (Most Features)**
```bash
# Use conda for RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia cuml
```

**Option D: CuPy (Memory Efficient)**
```bash
pip install cupy-cuda11x  # or cupy-cuda12x
```

## Quick Example

```python
from gpu_clustering_raster import GPURasterClustering
import numpy as np

# 1. Load or create your raster data
# Shape should be (n_bands, n_rows, n_cols)
raster = np.random.randn(99, 2000, 4000).astype(np.float32)

# 2. Initialize clustering
clusterer = GPURasterClustering(raster)

# 3. Run clustering (choose one method)

# Method A: PyTorch (works with basic GPU)
labels, model = clusterer.method_4_pytorch_kmeans(n_clusters=10)

# Method B: Faiss (fastest)
labels, model = clusterer.method_6_faiss_gpu_kmeans(n_clusters=10)

# Method C: RAPIDS (most comprehensive)
labels, model = clusterer.method_1_rapids_kmeans(n_clusters=10)

# 4. Use the results
print(f"Clustered into {len(np.unique(labels))} groups")
print(f"Result shape: {labels.shape}")  # (2000, 4000)

# 5. Save results
np.save('cluster_labels.npy', labels)
```

## Which Method Should I Use?

| If you want... | Use this method |
|----------------|----------------|
| Fastest performance | `method_6_faiss_gpu_kmeans()` |
| Easiest installation | `method_4_pytorch_kmeans()` |
| Density-based clustering | `method_2_rapids_dbscan()` |
| Memory efficiency | `method_5_cupy_minibatch_kmeans()` |
| Auto cluster detection | `method_3_rapids_hdbscan()` |
| Production stability | `method_1_rapids_kmeans()` |

## Test Your Installation

```bash
# Run the example script
python example_usage.py
```

## Visualization

```python
from visualization import visualize_clusters

# Visualize your results
visualize_clusters(labels, "My Clustering Results", save_path="clusters.png")
```

## Common Parameters

### KMeans-based methods (1, 4, 5, 6)
- `n_clusters`: Number of clusters (try 5-15 for most cases)
- `max_iter`: Max iterations (default 300 is usually fine)

### DBSCAN (method 2)
- `eps`: Neighborhood radius (try 0.3-1.0)
- `min_samples`: Min points per cluster (try 5-10)

### HDBSCAN (method 3)
- `min_cluster_size`: Min cluster size (try 50-200)
- `min_samples`: Min samples (try 5-10)

## Troubleshooting

**Out of memory?**
- Use `method_5_cupy_minibatch_kmeans()` with smaller batch_size
- Reduce raster size
- Use float16 instead of float32

**Too slow?**
- Use `method_6_faiss_gpu_kmeans()` (fastest)
- Reduce max_iter
- Use smaller data sample

**ImportError?**
- Check if GPU libraries are installed
- Verify CUDA version matches library
- Try method_4 (PyTorch) as fallback

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Check [CLUSTERING_GUIDE.md](CLUSTERING_GUIDE.md) for method comparisons
3. Run [benchmark_comparison.py](benchmark_comparison.py) to compare methods
4. Explore [visualization.py](visualization.py) for result visualization

## Real Data Example

```python
import rasterio
from gpu_clustering_raster import GPURasterClustering

# Load real satellite imagery
with rasterio.open('satellite.tif') as src:
    raster = src.read()  # (bands, rows, cols)

# Cluster
clusterer = GPURasterClustering(raster)
labels, model = clusterer.method_6_faiss_gpu_kmeans(n_clusters=8)

# Save with georeference
with rasterio.open(
    'clusters.tif', 'w',
    driver='GTiff',
    height=labels.shape[0],
    width=labels.shape[1],
    count=1,
    dtype=labels.dtype,
    crs=src.crs,
    transform=src.transform
) as dst:
    dst.write(labels, 1)
```

## Performance Tips

1. **Use float32** (not float64) for better GPU performance
2. **Normalize data** if features have different scales  
3. **Start small** - test with subset before full raster
4. **Batch processing** for multiple rasters
5. **Profile your code** to identify bottlenecks

## Getting Help

- Check the [CLUSTERING_GUIDE.md](CLUSTERING_GUIDE.md) for detailed explanations
- Review example scripts in the repository
- Ensure GPU drivers and CUDA are properly installed
- Verify GPU is detected: `nvidia-smi`

Happy clustering! 🚀

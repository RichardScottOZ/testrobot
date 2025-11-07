# GPU-Accelerated Clustering Methods Guide

## Overview

This document provides detailed information about six GPU-accelerated clustering algorithms designed for raster stack data with shape (99, 2000, 4000).

## Data Specifications

- **Input Shape**: (99, 2000, 4000)
  - 99 spectral bands/features
  - 2000 rows × 4000 columns
  - 8,000,000 total pixels
  - ~3.2 GB as float32

- **Processing**: Internally reshaped to (8,000,000, 99) for clustering
- **Output**: Cluster labels in shape (2000, 4000)

## Method Descriptions

### Method 1: RAPIDS cuML KMeans

**Algorithm**: K-Means clustering  
**Library**: NVIDIA RAPIDS cuML  
**GPU Optimization**: CUDA-accelerated Lloyd's algorithm

**Characteristics**:
- Partitioning-based clustering
- Requires pre-specified number of clusters
- Minimizes within-cluster variance
- Iterative refinement of centroids

**When to Use**:
- Known number of clusters
- Spherical, similar-sized clusters expected
- Need for fast, scalable solution
- Production environments

**Parameters**:
- `n_clusters`: Number of clusters (e.g., 5-20 for land cover)
- `max_iter`: Maximum iterations (default: 300)
- `tol`: Convergence tolerance

**Advantages**:
- Extremely fast on GPU
- Scalable to millions of samples
- Production-ready and stable
- Integrates with RAPIDS ecosystem

**Limitations**:
- Assumes spherical clusters
- Sensitive to initialization
- Requires knowing cluster count
- May converge to local optima

**Typical Use Cases**:
- Land cover classification
- Image segmentation
- Unsupervised classification
- Feature quantization

---

### Method 2: RAPIDS cuML DBSCAN

**Algorithm**: Density-Based Spatial Clustering of Applications with Noise  
**Library**: NVIDIA RAPIDS cuML  
**GPU Optimization**: CUDA-accelerated density estimation

**Characteristics**:
- Density-based clustering
- Discovers arbitrary-shaped clusters
- Identifies noise/outliers
- No need to specify cluster count

**When to Use**:
- Unknown number of clusters
- Non-spherical cluster shapes
- Need to identify outliers
- Spatial data with varying densities

**Parameters**:
- `eps`: Neighborhood radius (critical parameter)
- `min_samples`: Minimum points for core point
- `metric`: Distance metric (default: Euclidean)

**Advantages**:
- Handles arbitrary shapes
- Robust to outliers
- No cluster count needed
- Identifies noise points

**Limitations**:
- Sensitive to parameters (eps, min_samples)
- Struggles with varying densities
- Computationally intensive
- May label many points as noise

**Typical Use Cases**:
- Anomaly detection in imagery
- Finding spatial patterns
- Removing noise from data
- Urban area detection

---

### Method 3: RAPIDS cuML HDBSCAN

**Algorithm**: Hierarchical DBSCAN  
**Library**: NVIDIA RAPIDS cuML  
**GPU Optimization**: GPU-accelerated hierarchy building

**Characteristics**:
- Hierarchical density-based clustering
- Automatic cluster detection
- Handles varying densities
- More robust than DBSCAN

**When to Use**:
- Varying cluster densities
- Hierarchical structure in data
- Need automatic cluster detection
- Complex spatial patterns

**Parameters**:
- `min_cluster_size`: Minimum cluster size
- `min_samples`: Minimum samples for core point
- `cluster_selection_epsilon`: Optional distance threshold
- `gen_min_span_tree`: Generate minimum spanning tree

**Advantages**:
- Automatic cluster count
- Handles varying densities
- More robust than DBSCAN
- Provides cluster hierarchy

**Limitations**:
- Computationally expensive
- More complex parameters
- Longer execution time
- Higher memory usage

**Typical Use Cases**:
- Complex landscape classification
- Multi-scale analysis
- Hierarchical land cover mapping
- Automated scene understanding

---

### Method 4: PyTorch KMeans

**Algorithm**: K-Means (custom implementation)  
**Library**: PyTorch  
**GPU Optimization**: PyTorch CUDA tensors

**Characteristics**:
- Custom K-Means in PyTorch
- Integrates with deep learning
- Flexible and customizable
- Differentiable (if needed)

**When to Use**:
- Integration with PyTorch pipeline
- Need for customization
- Combining with neural networks
- Research applications

**Parameters**:
- `n_clusters`: Number of clusters
- `max_iter`: Maximum iterations
- Custom parameters possible

**Advantages**:
- Full control over algorithm
- Integrates with PyTorch
- Can be customized
- Potential for differentiable clustering

**Limitations**:
- Custom implementation
- May be slower than optimized libraries
- Requires PyTorch expertise
- Less optimized than specialized tools

**Typical Use Cases**:
- Deep learning workflows
- End-to-end trainable systems
- Research and experimentation
- Custom clustering variants

---

### Method 5: CuPy MiniBatch KMeans

**Algorithm**: Mini-Batch K-Means  
**Library**: CuPy (custom implementation)  
**GPU Optimization**: CuPy GPU arrays

**Characteristics**:
- Processes data in batches
- Memory-efficient
- Approximate K-Means
- Scalable to very large data

**When to Use**:
- Data larger than GPU memory
- Need memory efficiency
- Can accept approximate results
- Very large rasters

**Parameters**:
- `n_clusters`: Number of clusters
- `batch_size`: Mini-batch size (tune for GPU memory)
- `max_iter`: Number of iterations

**Advantages**:
- Very memory efficient
- Handles large datasets
- Faster than full K-Means
- Scalable

**Limitations**:
- Approximate results
- Requires batch size tuning
- May need more iterations
- Less accurate than full K-Means

**Typical Use Cases**:
- Very large rasters (> GPU memory)
- Quick exploratory analysis
- Real-time processing
- Streaming data

---

### Method 6: Faiss GPU KMeans

**Algorithm**: K-Means  
**Library**: Facebook AI Similarity Search (Faiss)  
**GPU Optimization**: Highly optimized CUDA kernels

**Characteristics**:
- State-of-the-art implementation
- Extremely optimized
- Focus on speed
- Production-grade

**When to Use**:
- Need maximum speed
- Production deployment
- Large-scale processing
- Batch processing many rasters

**Parameters**:
- `n_clusters`: Number of clusters
- `max_iter`: Maximum iterations
- `nredo`: Number of runs (best kept)
- `gpu`: GPU device selection

**Advantages**:
- Fastest K-Means available
- Highly optimized
- Production-tested
- Multi-GPU support

**Limitations**:
- K-Means only (no DBSCAN, etc.)
- Less flexible parameters
- Requires Faiss installation
- Focused on speed over flexibility

**Typical Use Cases**:
- Production pipelines
- Large-scale batch processing
- Speed-critical applications
- Similarity search with clustering

---

## Comparison Matrix

| Feature | RAPIDS KMeans | RAPIDS DBSCAN | RAPIDS HDBSCAN | PyTorch KMeans | CuPy MiniBatch | Faiss KMeans |
|---------|--------------|---------------|----------------|----------------|----------------|--------------|
| **Speed** | Fast | Medium | Slow | Medium | Fast | Fastest |
| **Memory** | Medium | High | High | Medium | Low | Medium |
| **Cluster Shape** | Spherical | Arbitrary | Arbitrary | Spherical | Spherical | Spherical |
| **Auto Clusters** | No | Yes | Yes | No | No | No |
| **Outlier Detection** | No | Yes | Yes | No | No | No |
| **Scalability** | High | Medium | Medium | High | Very High | Very High |
| **Flexibility** | Medium | Medium | Medium | High | High | Low |
| **Production Ready** | Yes | Yes | Yes | No | No | Yes |

## Parameter Selection Guidelines

### For KMeans-based methods (1, 4, 5, 6):

**Number of Clusters**:
- Land cover: 5-15 clusters
- Urban analysis: 8-20 clusters
- Vegetation mapping: 6-12 clusters
- General purpose: Start with 10

**Convergence**:
- `max_iter=300`: Usually sufficient
- Increase if not converging
- Monitor convergence in output

### For DBSCAN (Method 2):

**eps (neighborhood radius)**:
- Start with mean k-nearest neighbor distance
- Too small: Many noise points
- Too large: Few large clusters
- Typical range: 0.3-1.0 for normalized data

**min_samples**:
- Higher = more noise, fewer clusters
- Lower = more clusters, less noise
- Typical: 5-10 for most data
- Use 2*n_features as rule of thumb

### For HDBSCAN (Method 3):

**min_cluster_size**:
- Minimum meaningful cluster size
- Consider application requirements
- Typical: 50-200 for raster data

**min_samples**:
- Controls density threshold
- Similar to DBSCAN
- Typical: 5-10

## Memory Requirements

### GPU Memory Estimates for (99, 2000, 4000) raster:

- **Input data**: ~3.2 GB (float32)
- **Working memory**: 2-4 GB
- **Total required**: 6-8 GB

**Recommendations**:
- Minimum: 8 GB VRAM
- Recommended: 12 GB+ VRAM
- Optimal: 24 GB+ VRAM

### For Larger Rasters:

**If data > GPU memory**:
1. Use Method 5 (MiniBatch)
2. Use tiled processing
3. Sample data first, then predict
4. Reduce precision (float16)

## Performance Optimization Tips

1. **Data Preparation**:
   - Use float32 (not float64)
   - Ensure contiguous memory
   - Normalize features if needed

2. **Batch Size Tuning**:
   - Start with 10,000 for MiniBatch
   - Increase until memory saturated
   - Monitor GPU utilization

3. **Multi-GPU**:
   - Use RAPIDS Dask for multi-GPU
   - Partition data across GPUs
   - Aggregate results

4. **Precision**:
   - float32 usually sufficient
   - float16 for memory-constrained
   - Minimal accuracy loss

## Workflow Recommendations

### Exploratory Analysis:
1. Start with small sample
2. Try Method 4 (PyTorch) or Method 5 (MiniBatch)
3. Visualize results
4. Refine parameters

### Production Pipeline:
1. Benchmark all methods
2. Choose Method 1 (RAPIDS) or Method 6 (Faiss)
3. Validate on test data
4. Deploy with monitoring

### Research:
1. Use Method 4 (PyTorch) for flexibility
2. Experiment with custom variants
3. Compare with Methods 1-3
4. Publish reproducible code

## Troubleshooting

### Common Issues:

**Out of Memory**:
- Reduce batch size
- Use Method 5 (MiniBatch)
- Process in tiles
- Use float16

**Slow Performance**:
- Check GPU utilization
- Verify CUDA is used
- Reduce data size
- Optimize parameters

**Poor Clustering**:
- Normalize data
- Try different methods
- Adjust parameters
- Check for outliers

**Installation Issues**:
- Use conda for RAPIDS
- Match CUDA versions
- Check GPU compatibility
- Verify driver versions

## References

1. RAPIDS cuML: https://docs.rapids.ai/api/cuml/stable/
2. PyTorch: https://pytorch.org/docs/
3. CuPy: https://docs.cupy.dev/
4. Faiss: https://github.com/facebookresearch/faiss/wiki
5. DBSCAN: Ester et al., 1996
6. HDBSCAN: Campello et al., 2013

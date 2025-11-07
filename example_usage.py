"""
Example usage of the three drillhole visualization algorithms.

This demonstrates how to use each algorithm to visualize drillholes on a 3D globe.
"""

import numpy as np
from drillhole_visualization import Drillhole, VisualizationQuality
from algorithm_lod import LODVisualizationAlgorithm
from algorithm_clustering import ClusteringVisualizationAlgorithm
from algorithm_gpu import GPUInstancedVisualizationAlgorithm


def generate_sample_drillholes(num_drillholes: int = 100000) -> list:
    """
    Generate sample drillhole data for testing.
    
    Args:
        num_drillholes: Number of drillholes to generate
        
    Returns:
        List of Drillhole objects
    """
    print(f"Generating {num_drillholes:,} sample drillholes...")
    
    np.random.seed(42)
    drillholes = []
    
    for i in range(num_drillholes):
        # Random locations around the globe
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        
        # Typical drillhole depths (100m to 5000m)
        depth = np.random.uniform(100, 5000)
        
        # Standard drillhole diameter
        diameter = np.random.choice([0.076, 0.096, 0.122])  # Common sizes in meters
        
        drillhole = Drillhole(
            latitude=lat,
            longitude=lon,
            depth=depth,
            diameter=diameter,
            id=i
        )
        drillholes.append(drillhole)
    
    print(f"Generated {len(drillholes):,} drillholes")
    return drillholes


def demo_algorithm_1_lod():
    """Demonstrate Algorithm 1: LOD with Octree."""
    print("\n" + "="*70)
    print("ALGORITHM 1: Level of Detail with Octree Spatial Indexing")
    print("="*70)
    
    # Generate sample data
    drillholes = generate_sample_drillholes(10000)
    
    # Initialize algorithm
    algorithm = LODVisualizationAlgorithm(exaggeration_factor=500.0)
    
    # Prepare data
    algorithm.prepare_data(drillholes)
    
    # Print statistics
    stats = algorithm.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Simulate camera positions
    camera_positions = [
        (7000000, 0, 0),          # 1000 km above equator
        (10000000, 0, 0),         # Far from Earth
        (6500000, 0, 0),          # Close to surface
    ]
    
    for quality in [VisualizationQuality.LOW, VisualizationQuality.MEDIUM, 
                    VisualizationQuality.HIGH, VisualizationQuality.ULTRA]:
        print(f"\n{quality.name} Quality:")
        
        for i, cam_pos in enumerate(camera_positions):
            cam_dir = (-cam_pos[0], -cam_pos[1], -cam_pos[2])  # Look toward origin
            
            visible = algorithm.get_visible_drillholes(cam_pos, cam_dir, quality)
            print(f"  Camera {i+1}: {len(visible):,} visible drillholes")


def demo_algorithm_2_clustering():
    """Demonstrate Algorithm 2: Hierarchical Clustering."""
    print("\n" + "="*70)
    print("ALGORITHM 2: Clustering-based Aggregation")
    print("="*70)
    
    # Generate sample data
    drillholes = generate_sample_drillholes(5000)  # Smaller for faster clustering
    
    # Initialize algorithm
    algorithm = ClusteringVisualizationAlgorithm(
        exaggeration_factor=500.0,
        num_levels=4
    )
    
    # Prepare data
    algorithm.prepare_data(drillholes)
    
    # Print statistics
    stats = algorithm.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Simulate camera positions
    camera_positions = [
        (7000000, 0, 0),
        (10000000, 0, 0),
        (6500000, 0, 0),
    ]
    
    for quality in [VisualizationQuality.LOW, VisualizationQuality.HIGH]:
        print(f"\n{quality.name} Quality:")
        
        for i, cam_pos in enumerate(camera_positions):
            cam_dir = (-cam_pos[0], -cam_pos[1], -cam_pos[2])
            
            visible = algorithm.get_visible_drillholes(cam_pos, cam_dir, quality)
            print(f"  Camera {i+1}: {len(visible):,} drillholes/clusters to render")


def demo_algorithm_3_gpu():
    """Demonstrate Algorithm 3: GPU Instanced Rendering."""
    print("\n" + "="*70)
    print("ALGORITHM 3: GPU-Accelerated Instanced Rendering")
    print("="*70)
    
    # Generate sample data
    drillholes = generate_sample_drillholes(50000)
    
    # Initialize algorithm
    algorithm = GPUInstancedVisualizationAlgorithm(exaggeration_factor=500.0)
    
    # Prepare data
    algorithm.prepare_data(drillholes)
    
    # Print statistics
    stats = algorithm.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get instance buffer info
    instance_buffer = algorithm.get_instance_buffer()
    print(f"\nInstance buffer shape: {instance_buffer.shape}")
    print(f"Instance buffer dtype: {instance_buffer.dtype}")
    print(f"First instance data: {instance_buffer[0]}")
    
    # Generate shader code
    shaders = algorithm.generate_shader_code()
    print(f"\nGenerated shaders:")
    print(f"  Vertex shader: {len(shaders['vertex'])} characters")
    print(f"  Fragment shader: {len(shaders['fragment'])} characters")
    print(f"  Compute shader: {len(shaders['compute'])} characters")
    
    # Test visibility culling
    camera_positions = [
        (7000000, 0, 0),
        (10000000, 0, 0),
    ]
    
    for quality in [VisualizationQuality.MEDIUM, VisualizationQuality.ULTRA]:
        print(f"\n{quality.name} Quality:")
        
        for i, cam_pos in enumerate(camera_positions):
            cam_dir = (-cam_pos[0], -cam_pos[1], -cam_pos[2])
            
            visible = algorithm.get_visible_drillholes(cam_pos, cam_dir, quality)
            print(f"  Camera {i+1}: {len(visible):,} instances to render")


def comparison_benchmark():
    """Compare performance characteristics of all three algorithms."""
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    
    # Use same dataset for all algorithms
    print("\nTesting with 10,000 drillholes...")
    drillholes = generate_sample_drillholes(10000)
    
    algorithms = [
        ("LOD + Octree", LODVisualizationAlgorithm(500.0)),
        ("Clustering", ClusteringVisualizationAlgorithm(500.0, num_levels=3)),
        ("GPU Instanced", GPUInstancedVisualizationAlgorithm(500.0))
    ]
    
    # Prepare all algorithms
    print("\nPreparing algorithms...")
    for name, algo in algorithms:
        print(f"\n{name}:")
        algo.prepare_data(drillholes)
        stats = algo.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Test rendering
    print("\n\nRendering test (camera at 1000km altitude):")
    camera_pos = (7371000, 0, 0)
    camera_dir = (-1, 0, 0)
    
    for name, algo in algorithms:
        visible = algo.get_visible_drillholes(
            camera_pos, 
            camera_dir, 
            VisualizationQuality.MEDIUM
        )
        print(f"{name}: {len(visible):,} items to render")


if __name__ == "__main__":
    print("3D Global Drillhole Visualization - Algorithm Demonstrations\n")
    
    # Run all demonstrations
    demo_algorithm_1_lod()
    demo_algorithm_2_clustering()
    demo_algorithm_3_gpu()
    comparison_benchmark()
    
    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70)

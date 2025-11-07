"""
Unit tests for drillhole visualization algorithms.
"""

import unittest
import numpy as np
from drillhole_visualization import Drillhole, VisualizationQuality
from algorithm_lod import LODVisualizationAlgorithm, OctreeNode
from algorithm_clustering import ClusteringVisualizationAlgorithm, DrillholeCluster
from algorithm_gpu import GPUInstancedVisualizationAlgorithm


class TestDrillhole(unittest.TestCase):
    """Test Drillhole class."""
    
    def test_creation(self):
        """Test creating a drillhole."""
        dh = Drillhole(latitude=45.0, longitude=-120.0, depth=1000.0, diameter=0.1)
        self.assertEqual(dh.latitude, 45.0)
        self.assertEqual(dh.longitude, -120.0)
        self.assertEqual(dh.depth, 1000.0)
        self.assertEqual(dh.diameter, 0.1)
    
    def test_to_cartesian(self):
        """Test conversion to Cartesian coordinates."""
        # Test equator
        dh = Drillhole(latitude=0.0, longitude=0.0, depth=100.0)
        x, y, z = dh.to_cartesian()
        self.assertAlmostEqual(y, 0.0, places=1)
        self.assertAlmostEqual(z, 0.0, places=1)
        self.assertGreater(x, 6000000)  # Should be near Earth radius
        
        # Test north pole
        dh = Drillhole(latitude=90.0, longitude=0.0, depth=100.0)
        x, y, z = dh.to_cartesian()
        self.assertAlmostEqual(x, 0.0, places=1)
        self.assertAlmostEqual(y, 0.0, places=1)
        self.assertGreater(z, 6000000)
    
    def test_bounding_box(self):
        """Test bounding box calculation."""
        dh = Drillhole(latitude=0.0, longitude=0.0, depth=1000.0, diameter=0.1)
        min_corner, max_corner = dh.get_bounding_box()
        
        self.assertEqual(len(min_corner), 3)
        self.assertEqual(len(max_corner), 3)
        
        # Max should be greater than min in all dimensions
        for i in range(3):
            self.assertGreater(max_corner[i], min_corner[i])


class TestOctree(unittest.TestCase):
    """Test Octree spatial indexing."""
    
    def test_octree_creation(self):
        """Test creating an octree."""
        octree = OctreeNode((-100, -100, -100), (100, 100, 100))
        self.assertTrue(octree.is_leaf)
        self.assertEqual(len(octree.drillholes), 0)
    
    def test_octree_insert(self):
        """Test inserting drillholes into octree."""
        octree = OctreeNode((-10000000, -10000000, -10000000), 
                           (10000000, 10000000, 10000000),
                           max_drillholes=10)
        
        # Insert some drillholes
        drillholes = [
            Drillhole(0.0, 0.0, 100.0, id=i) for i in range(5)
        ]
        
        for dh in drillholes:
            result = octree.insert(dh)
            self.assertTrue(result)
        
        self.assertEqual(len(octree.drillholes), 5)
    
    def test_octree_split(self):
        """Test octree splitting when capacity exceeded."""
        octree = OctreeNode((-10000000, -10000000, -10000000), 
                           (10000000, 10000000, 10000000),
                           max_drillholes=5,
                           max_depth=2)
        
        # Insert enough drillholes to trigger split
        for i in range(10):
            dh = Drillhole(i * 10.0, i * 10.0, 100.0, id=i)
            octree.insert(dh)
        
        # Should have split into children
        self.assertFalse(octree.is_leaf)
        self.assertIsNotNone(octree.children)
        self.assertEqual(len(octree.children), 8)


class TestLODAlgorithm(unittest.TestCase):
    """Test LOD visualization algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algo = LODVisualizationAlgorithm(exaggeration_factor=1000.0)
        self.assertEqual(algo.exaggeration_factor, 1000.0)
        self.assertIsNone(algo.octree)
    
    def test_prepare_data(self):
        """Test data preparation."""
        algo = LODVisualizationAlgorithm()
        drillholes = [
            Drillhole(i * 10.0, i * 10.0, 100.0 + i, id=i) 
            for i in range(100)
        ]
        
        algo.prepare_data(drillholes)
        
        self.assertIsNotNone(algo.octree)
        self.assertEqual(algo.total_drillholes, 100)
    
    def test_get_visible_drillholes(self):
        """Test getting visible drillholes."""
        algo = LODVisualizationAlgorithm()
        drillholes = [
            Drillhole(0.0, 0.0, 100.0 + i, id=i) 
            for i in range(50)
        ]
        
        algo.prepare_data(drillholes)
        
        camera_pos = (7000000, 0, 0)  # 1000 km above surface
        camera_dir = (-1, 0, 0)  # Looking toward Earth
        
        visible = algo.get_visible_drillholes(
            camera_pos, 
            camera_dir, 
            VisualizationQuality.MEDIUM
        )
        
        # Should get some visible drillholes
        self.assertIsInstance(visible, list)
    
    def test_statistics(self):
        """Test getting statistics."""
        algo = LODVisualizationAlgorithm()
        drillholes = [Drillhole(0.0, 0.0, 100.0, id=i) for i in range(10)]
        
        algo.prepare_data(drillholes)
        stats = algo.get_statistics()
        
        self.assertIn('algorithm', stats)
        self.assertIn('total_drillholes', stats)
        self.assertEqual(stats['total_drillholes'], 10)


class TestClusteringAlgorithm(unittest.TestCase):
    """Test clustering visualization algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algo = ClusteringVisualizationAlgorithm(
            exaggeration_factor=500.0,
            num_levels=3
        )
        self.assertEqual(algo.exaggeration_factor, 500.0)
        self.assertEqual(algo.num_levels, 3)
    
    def test_cluster_creation(self):
        """Test creating a cluster."""
        drillholes = [
            Drillhole(0.0, 0.0, 100.0, id=0),
            Drillhole(0.1, 0.1, 150.0, id=1),
        ]
        
        cluster = DrillholeCluster(
            center_lat=0.05,
            center_lon=0.05,
            avg_depth=125.0,
            drillhole_count=2,
            drillholes=drillholes
        )
        
        self.assertEqual(cluster.drillhole_count, 2)
        self.assertIsNotNone(cluster.representative)
    
    def test_prepare_data(self):
        """Test data preparation with clustering."""
        algo = ClusteringVisualizationAlgorithm(num_levels=2)
        
        drillholes = [
            Drillhole(i * 5.0, i * 5.0, 100.0 + i, id=i) 
            for i in range(100)
        ]
        
        algo.prepare_data(drillholes)
        
        self.assertEqual(len(algo.all_drillholes), 100)
        self.assertGreater(len(algo.clusters), 0)
    
    def test_get_visible_drillholes(self):
        """Test getting visible drillholes with clustering."""
        algo = ClusteringVisualizationAlgorithm(num_levels=2)
        drillholes = [
            Drillhole(0.0, 0.0, 100.0, id=i) 
            for i in range(20)
        ]
        
        algo.prepare_data(drillholes)
        
        camera_pos = (7000000, 0, 0)
        camera_dir = (-1, 0, 0)
        
        visible = algo.get_visible_drillholes(
            camera_pos,
            camera_dir,
            VisualizationQuality.MEDIUM
        )
        
        self.assertIsInstance(visible, list)


class TestGPUAlgorithm(unittest.TestCase):
    """Test GPU instanced rendering algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        algo = GPUInstancedVisualizationAlgorithm(exaggeration_factor=800.0)
        self.assertEqual(algo.exaggeration_factor, 800.0)
        self.assertIsNone(algo.instance_data)
    
    def test_prepare_data(self):
        """Test preparing instance data."""
        algo = GPUInstancedVisualizationAlgorithm()
        drillholes = [
            Drillhole(i * 10.0, i * 10.0, 100.0 + i, diameter=0.1, id=i) 
            for i in range(50)
        ]
        
        algo.prepare_data(drillholes)
        
        self.assertIsNotNone(algo.instance_data)
        self.assertEqual(algo.instance_data.shape[0], 50)
        self.assertEqual(algo.instance_data.shape[1], algo.instance_stride)
    
    def test_instance_buffer(self):
        """Test getting instance buffer."""
        algo = GPUInstancedVisualizationAlgorithm()
        drillholes = [Drillhole(0.0, 0.0, 100.0, id=0)]
        
        algo.prepare_data(drillholes)
        buffer = algo.get_instance_buffer()
        
        self.assertIsInstance(buffer, np.ndarray)
        self.assertEqual(buffer.dtype, np.float32)
    
    def test_shader_generation(self):
        """Test shader code generation."""
        algo = GPUInstancedVisualizationAlgorithm()
        shaders = algo.generate_shader_code()
        
        self.assertIn('vertex', shaders)
        self.assertIn('fragment', shaders)
        self.assertIn('compute', shaders)
        
        # Check that shaders contain expected code
        self.assertIn('gl_Position', shaders['vertex'])
        self.assertIn('out_color', shaders['fragment'])
        self.assertIn('visibility', shaders['compute'])
    
    def test_get_visible_drillholes(self):
        """Test CPU-side visibility culling."""
        algo = GPUInstancedVisualizationAlgorithm()
        drillholes = [
            Drillhole(0.0, 0.0, 100.0, id=i) 
            for i in range(30)
        ]
        
        algo.prepare_data(drillholes)
        
        camera_pos = (7000000, 0, 0)
        camera_dir = (-1, 0, 0)
        
        visible = algo.get_visible_drillholes(
            camera_pos,
            camera_dir,
            VisualizationQuality.HIGH
        )
        
        self.assertIsInstance(visible, list)


class TestRenderGeometry(unittest.TestCase):
    """Test geometry generation for rendering."""
    
    def test_render_geometry(self):
        """Test generating pipe geometry."""
        algo = LODVisualizationAlgorithm()
        drillhole = Drillhole(0.0, 0.0, 1000.0, diameter=0.1)
        
        geometry = algo.get_render_geometry(drillhole)
        
        self.assertIn('vertices', geometry)
        self.assertIn('normals', geometry)
        self.assertIn('indices', geometry)
        
        # Check array shapes
        vertices = geometry['vertices']
        normals = geometry['normals']
        indices = geometry['indices']
        
        self.assertEqual(len(vertices.shape), 2)
        self.assertEqual(vertices.shape[1], 3)  # x, y, z
        self.assertEqual(vertices.shape, normals.shape)
        self.assertGreater(len(indices), 0)


if __name__ == '__main__':
    unittest.main()

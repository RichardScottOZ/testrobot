"""
Algorithm 1: Level of Detail (LOD) Rendering with Spatial Indexing

This algorithm uses an octree spatial data structure to organize drillholes
and implements level-of-detail rendering to handle hundreds of millions of
drillholes efficiently.

Key features:
- Octree spatial indexing for fast culling
- Dynamic LOD based on distance from camera
- Frustum culling to only render visible drillholes
- Memory-efficient representation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from drillhole_visualization import (
    Drillhole, BaseVisualizationAlgorithm, VisualizationQuality
)


class OctreeNode:
    """Node in the octree spatial data structure."""
    
    def __init__(self, 
                 min_corner: Tuple[float, float, float],
                 max_corner: Tuple[float, float, float],
                 max_depth: int = 10,
                 max_drillholes: int = 100):
        """
        Initialize octree node.
        
        Args:
            min_corner: Minimum corner of bounding box (x, y, z)
            max_corner: Maximum corner of bounding box (x, y, z)
            max_depth: Maximum depth of octree
            max_drillholes: Maximum drillholes per leaf node before splitting
        """
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.center = (self.min_corner + self.max_corner) / 2
        self.size = self.max_corner - self.min_corner
        
        self.max_depth = max_depth
        self.max_drillholes = max_drillholes
        
        self.drillholes: List[Drillhole] = []
        self.children: Optional[List['OctreeNode']] = None
        self.is_leaf = True
        
    def insert(self, drillhole: Drillhole, current_depth: int = 0) -> bool:
        """Insert a drillhole into the octree."""
        # Check if drillhole is within this node's bounds
        min_bbox, max_bbox = drillhole.get_bounding_box()
        
        if not self._intersects_bbox(min_bbox, max_bbox):
            return False
        
        # If this is a leaf and we have room, add it here
        if self.is_leaf:
            self.drillholes.append(drillhole)
            
            # Split if we exceed capacity and haven't reached max depth
            if len(self.drillholes) > self.max_drillholes and current_depth < self.max_depth:
                self._split(current_depth)
            
            return True
        
        # Otherwise, insert into appropriate child
        inserted = False
        for child in self.children:
            if child.insert(drillhole, current_depth + 1):
                inserted = True
        
        return inserted
    
    def _intersects_bbox(self, min_bbox: Tuple[float, float, float], 
                        max_bbox: Tuple[float, float, float]) -> bool:
        """Check if a bounding box intersects with this node."""
        min_bbox = np.array(min_bbox)
        max_bbox = np.array(max_bbox)
        
        # AABB intersection test
        return np.all(min_bbox <= self.max_corner) and np.all(max_bbox >= self.min_corner)
    
    def _split(self, current_depth: int):
        """Split this node into 8 children."""
        self.is_leaf = False
        self.children = []
        
        # Create 8 octants
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_min = self.min_corner + self.size / 2 * np.array([i, j, k])
                    child_max = child_min + self.size / 2
                    
                    child = OctreeNode(
                        tuple(child_min),
                        tuple(child_max),
                        self.max_depth,
                        self.max_drillholes
                    )
                    self.children.append(child)
        
        # Redistribute drillholes to children
        for drillhole in self.drillholes:
            for child in self.children:
                child.insert(drillhole, current_depth + 1)
        
        # Clear drillholes from this node (now stored in children)
        self.drillholes = []
    
    def query_frustum(self, 
                     camera_position: np.ndarray,
                     camera_direction: np.ndarray,
                     fov: float,
                     max_distance: float) -> List[Drillhole]:
        """
        Query drillholes visible in the camera frustum.
        
        Args:
            camera_position: Camera position (x, y, z)
            camera_direction: Camera direction vector (normalized)
            fov: Field of view in radians
            max_distance: Maximum viewing distance
            
        Returns:
            List of potentially visible drillholes
        """
        # Simple frustum culling using sphere test
        to_center = self.center - camera_position
        distance = np.linalg.norm(to_center)
        
        # If node is too far, cull it
        sphere_radius = np.linalg.norm(self.size) / 2
        if distance - sphere_radius > max_distance:
            return []
        
        # If node is behind camera, cull it
        if distance > 0:
            direction_to_node = to_center / distance
            dot_product = np.dot(camera_direction, direction_to_node)
            
            # Conservative culling
            angle_threshold = np.cos(fov / 2 + np.arcsin(sphere_radius / max(distance, sphere_radius)))
            if dot_product < angle_threshold:
                return []
        
        # If this is a leaf, return drillholes
        if self.is_leaf:
            return self.drillholes
        
        # Otherwise, recursively query children
        result = []
        for child in self.children:
            result.extend(child.query_frustum(camera_position, camera_direction, fov, max_distance))
        
        return result


class LODVisualizationAlgorithm(BaseVisualizationAlgorithm):
    """
    Level of Detail rendering algorithm using octree spatial indexing.
    
    This algorithm organizes drillholes in an octree for efficient spatial queries
    and implements distance-based LOD to render appropriate detail levels.
    """
    
    def __init__(self, exaggeration_factor: float = 1000.0):
        """Initialize LOD algorithm."""
        super().__init__(exaggeration_factor)
        self.octree: Optional[OctreeNode] = None
        self.total_drillholes = 0
        
        # LOD distance thresholds (in meters)
        self.lod_distances = {
            VisualizationQuality.ULTRA: 100000,    # 100 km
            VisualizationQuality.HIGH: 500000,     # 500 km
            VisualizationQuality.MEDIUM: 1000000,  # 1000 km
            VisualizationQuality.LOW: 5000000,     # 5000 km
        }
    
    def prepare_data(self, drillholes: List[Drillhole]) -> None:
        """
        Build octree from drillholes.
        
        Args:
            drillholes: List of all drillholes to visualize
        """
        self.total_drillholes = len(drillholes)
        
        # Create root octree node covering the entire Earth
        earth_size = self.earth_radius * 2.5  # Slightly larger to cover depth
        self.octree = OctreeNode(
            (-earth_size, -earth_size, -earth_size),
            (earth_size, earth_size, earth_size),
            max_depth=12,  # Deep enough for good spatial resolution
            max_drillholes=1000  # Tune based on memory constraints
        )
        
        # Insert all drillholes into octree
        print(f"Building octree for {len(drillholes)} drillholes...")
        for i, drillhole in enumerate(drillholes):
            if i % 1000000 == 0 and i > 0:
                print(f"  Inserted {i:,} drillholes...")
            self.octree.insert(drillhole)
        
        print(f"Octree built successfully with {len(drillholes):,} drillholes")
    
    def get_visible_drillholes(self,
                              camera_position: Tuple[float, float, float],
                              camera_direction: Tuple[float, float, float],
                              quality: VisualizationQuality) -> List[Drillhole]:
        """
        Get drillholes visible from camera position using LOD.
        
        Args:
            camera_position: Camera position in 3D space (x, y, z)
            camera_direction: Normalized camera direction vector
            quality: Desired rendering quality
            
        Returns:
            List of drillholes to render with appropriate LOD
        """
        if self.octree is None:
            return []
        
        camera_pos = np.array(camera_position)
        camera_dir = np.array(camera_direction)
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        # Get maximum viewing distance based on quality
        max_distance = self.lod_distances[quality]
        
        # Query octree for visible drillholes
        fov = np.radians(60)  # 60 degree field of view
        visible = self.octree.query_frustum(camera_pos, camera_dir, fov, max_distance)
        
        # Apply distance-based LOD filtering
        filtered = []
        for drillhole in visible:
            pos = np.array(drillhole.to_cartesian(self.earth_radius))
            distance = np.linalg.norm(pos - camera_pos)
            
            # Sample drillholes based on distance (keep fewer at greater distances)
            if quality == VisualizationQuality.ULTRA:
                keep_ratio = 1.0
            elif quality == VisualizationQuality.HIGH:
                keep_ratio = min(1.0, max_distance / (distance + 1))
            elif quality == VisualizationQuality.MEDIUM:
                keep_ratio = min(1.0, max_distance / (2 * distance + 1))
            else:  # LOW
                keep_ratio = min(1.0, max_distance / (4 * distance + 1))
            
            # Deterministic sampling based on drillhole id hash
            if drillhole.id is None:
                filtered.append(drillhole)
            else:
                # Use hash of ID for deterministic pseudo-random value in [0, 1)
                hash_value = (hash(drillhole.id) & 0xFFFFFFFF) / 0xFFFFFFFF
                if hash_value < keep_ratio:
                    filtered.append(drillhole)
        
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get statistics about the spatial index."""
        def count_nodes(node: OctreeNode) -> Tuple[int, int, int]:
            """Count nodes, leaves, and total drillholes in tree."""
            if node.is_leaf:
                return 1, 1, len(node.drillholes)
            
            total_nodes = 1
            total_leaves = 0
            total_drillholes = 0
            
            for child in node.children:
                n, l, d = count_nodes(child)
                total_nodes += n
                total_leaves += l
                total_drillholes += d
            
            return total_nodes, total_leaves, total_drillholes
        
        if self.octree is None:
            return {
                'algorithm': 'LOD with Octree',
                'total_drillholes': 0,
                'octree_nodes': 0,
                'octree_leaves': 0
            }
        
        nodes, leaves, drillholes = count_nodes(self.octree)
        
        return {
            'algorithm': 'LOD with Octree',
            'total_drillholes': self.total_drillholes,
            'octree_nodes': nodes,
            'octree_leaves': leaves,
            'avg_drillholes_per_leaf': drillholes / max(leaves, 1)
        }

"""
Algorithm 2: Clustering-based Aggregation with Dynamic Detail

This algorithm uses hierarchical clustering to group nearby drillholes and
dynamically adjusts the level of aggregation based on viewing distance.

Key features:
- K-means clustering at multiple hierarchical levels
- Dynamic aggregation based on camera distance
- Representative drillhole selection for distant clusters
- Smooth transitions between detail levels
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from drillhole_visualization import (
    Drillhole, BaseVisualizationAlgorithm, VisualizationQuality
)


@dataclass
class DrillholeCluster:
    """Represents a cluster of drillholes."""
    center_lat: float
    center_lon: float
    avg_depth: float
    drillhole_count: int
    drillholes: List[Drillhole]
    representative: Optional[Drillhole] = None
    subclusters: Optional[List['DrillholeCluster']] = None
    
    def __post_init__(self):
        """Initialize representative drillhole."""
        if self.representative is None and self.drillholes:
            # Use the drillhole closest to cluster center as representative
            min_dist = float('inf')
            for dh in self.drillholes:
                dist = ((dh.latitude - self.center_lat) ** 2 + 
                       (dh.longitude - self.center_lon) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    self.representative = dh
            
            # Create a representative with cluster average properties
            if self.representative:
                self.representative = Drillhole(
                    latitude=self.center_lat,
                    longitude=self.center_lon,
                    depth=self.avg_depth,
                    diameter=self.representative.diameter * np.sqrt(self.drillhole_count),
                    id=-1  # Cluster ID
                )


class ClusteringVisualizationAlgorithm(BaseVisualizationAlgorithm):
    """
    Clustering-based visualization algorithm.
    
    This algorithm pre-computes hierarchical clusters of drillholes and
    adaptively shows individual drillholes or cluster representatives
    based on viewing distance.
    """
    
    def __init__(self, exaggeration_factor: float = 1000.0, num_levels: int = 5):
        """
        Initialize clustering algorithm.
        
        Args:
            exaggeration_factor: Depth exaggeration for visibility
            num_levels: Number of hierarchical clustering levels
        """
        super().__init__(exaggeration_factor)
        self.num_levels = num_levels
        self.clusters: List[List[DrillholeCluster]] = []
        self.all_drillholes: List[Drillhole] = []
        
        # Distance thresholds for each clustering level (in meters)
        self.level_distances = [
            10000,      # Level 0: < 10 km - show individual drillholes
            100000,     # Level 1: < 100 km - small clusters
            500000,     # Level 2: < 500 km - medium clusters
            2000000,    # Level 3: < 2000 km - large clusters
            10000000,   # Level 4: < 10000 km - very large clusters
        ]
    
    def prepare_data(self, drillholes: List[Drillhole]) -> None:
        """
        Build hierarchical clusters from drillholes.
        
        Args:
            drillholes: List of all drillholes to visualize
        """
        self.all_drillholes = drillholes
        print(f"Building hierarchical clusters for {len(drillholes)} drillholes...")
        
        # Start with all drillholes as individual clusters
        current_clusters = [
            DrillholeCluster(
                center_lat=dh.latitude,
                center_lon=dh.longitude,
                avg_depth=dh.depth,
                drillhole_count=1,
                drillholes=[dh]
            ) for dh in drillholes
        ]
        
        # Build hierarchy from bottom up
        for level in range(self.num_levels):
            # Determine number of clusters for this level
            # Reduce by factor of ~10 at each level
            target_clusters = max(100, len(drillholes) // (10 ** (level + 1)))
            
            print(f"  Level {level}: Clustering {len(current_clusters)} clusters "
                  f"into ~{target_clusters} clusters...")
            
            # Perform clustering
            if len(current_clusters) <= target_clusters:
                # No need to cluster further
                self.clusters.append(current_clusters)
                current_clusters = current_clusters
            else:
                new_clusters = self._cluster_drillholes(current_clusters, target_clusters)
                self.clusters.append(new_clusters)
                current_clusters = new_clusters
        
        print(f"Hierarchical clustering complete: {len(self.clusters)} levels")
    
    def _cluster_drillholes(self, 
                           items: List[DrillholeCluster], 
                           num_clusters: int) -> List[DrillholeCluster]:
        """
        Cluster drillholes using k-means on lat/lon coordinates.
        
        Args:
            items: List of drillhole clusters to group
            num_clusters: Target number of clusters
            
        Returns:
            List of new clusters
        """
        if len(items) <= num_clusters:
            return items
        
        # Extract coordinates
        coords = np.array([[item.center_lat, item.center_lon] for item in items])
        
        # Simple k-means clustering
        # Initialize cluster centers randomly
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(coords), num_clusters, replace=False)
        centers = coords[indices].copy()
        
        max_iterations = 20
        for iteration in range(max_iterations):
            # Assign each point to nearest center
            distances = np.zeros((len(coords), num_clusters))
            for i in range(num_clusters):
                diff = coords - centers[i]
                # Use great circle distance approximation
                distances[:, i] = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
            
            assignments = np.argmin(distances, axis=1)
            
            # Update centers
            old_centers = centers.copy()
            for i in range(num_clusters):
                mask = assignments == i
                if np.any(mask):
                    centers[i] = coords[mask].mean(axis=0)
            
            # Check convergence
            if np.allclose(centers, old_centers):
                break
        
        # Create new clusters
        new_clusters = []
        for i in range(num_clusters):
            mask = assignments == i
            if not np.any(mask):
                continue
            
            cluster_items = [items[j] for j in range(len(items)) if mask[j]]
            
            # Aggregate drillholes
            all_drillholes = []
            for cluster in cluster_items:
                all_drillholes.extend(cluster.drillholes)
            
            if not all_drillholes:
                continue
            
            # Calculate cluster properties
            avg_lat = np.mean([dh.latitude for dh in all_drillholes])
            avg_lon = np.mean([dh.longitude for dh in all_drillholes])
            avg_depth = np.mean([dh.depth for dh in all_drillholes])
            
            new_cluster = DrillholeCluster(
                center_lat=avg_lat,
                center_lon=avg_lon,
                avg_depth=avg_depth,
                drillhole_count=len(all_drillholes),
                drillholes=all_drillholes,
                subclusters=cluster_items
            )
            
            new_clusters.append(new_cluster)
        
        return new_clusters
    
    def get_visible_drillholes(self,
                              camera_position: Tuple[float, float, float],
                              camera_direction: Tuple[float, float, float],
                              quality: VisualizationQuality) -> List[Drillhole]:
        """
        Get drillholes or cluster representatives based on distance.
        
        Args:
            camera_position: Camera position in 3D space (x, y, z)
            camera_direction: Normalized camera direction vector
            quality: Desired rendering quality
            
        Returns:
            List of drillholes to render (may include cluster representatives)
        """
        if not self.clusters:
            return []
        
        camera_pos = np.array(camera_position)
        
        # Adjust quality settings
        quality_multiplier = {
            VisualizationQuality.ULTRA: 2.0,
            VisualizationQuality.HIGH: 1.0,
            VisualizationQuality.MEDIUM: 0.5,
            VisualizationQuality.LOW: 0.25,
        }[quality]
        
        result = []
        
        # Determine which clustering level to use based on distance
        for cluster_list in self.clusters:
            for cluster in cluster_list:
                # Calculate distance to cluster center
                cluster_3d = Drillhole(
                    latitude=cluster.center_lat,
                    longitude=cluster.center_lon,
                    depth=0,
                    diameter=1
                ).to_cartesian(self.earth_radius)
                
                cluster_pos = np.array(cluster_3d)
                distance = np.linalg.norm(cluster_pos - camera_pos)
                
                # Determine if we should show individual drillholes or cluster
                threshold = self.level_distances[0] * quality_multiplier
                
                if distance < threshold and len(cluster.drillholes) < 1000:
                    # Show individual drillholes
                    result.extend(cluster.drillholes)
                elif cluster.representative:
                    # Show cluster representative
                    # Only if it's in viewing frustum (simplified check)
                    to_cluster = cluster_pos - camera_pos
                    to_cluster_norm = to_cluster / (np.linalg.norm(to_cluster) + 1e-6)
                    camera_dir = np.array(camera_direction)
                    camera_dir = camera_dir / (np.linalg.norm(camera_dir) + 1e-6)
                    
                    dot = np.dot(to_cluster_norm, camera_dir)
                    if dot > 0.3:  # Within ~70 degree FOV
                        result.append(cluster.representative)
        
        # Remove duplicates
        seen = set()
        unique_result = []
        for dh in result:
            key = (dh.latitude, dh.longitude, dh.depth)
            if key not in seen:
                seen.add(key)
                unique_result.append(dh)
        
        return unique_result
    
    def get_statistics(self) -> Dict:
        """Get statistics about the clustering."""
        if not self.clusters:
            return {
                'algorithm': 'Hierarchical Clustering',
                'total_drillholes': 0,
                'num_levels': 0,
                'clusters_per_level': []
            }
        
        return {
            'algorithm': 'Hierarchical Clustering',
            'total_drillholes': len(self.all_drillholes),
            'num_levels': len(self.clusters),
            'clusters_per_level': [len(level) for level in self.clusters],
            'max_cluster_size': max(c.drillhole_count for level in self.clusters for c in level),
            'avg_cluster_size': np.mean([c.drillhole_count for level in self.clusters for c in level])
        }

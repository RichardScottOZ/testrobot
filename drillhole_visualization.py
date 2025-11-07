"""
3D Global Drillhole Visualization Algorithms

This module provides three different algorithms for visualizing hundreds of millions
of drillholes as exaggerated pipes on a 3D globe.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Drillhole:
    """Represents a single drillhole with its position and properties."""
    latitude: float  # degrees
    longitude: float  # degrees
    depth: float  # meters
    diameter: float = 0.1  # meters
    id: Optional[int] = None
    
    def to_cartesian(self, earth_radius: float = 6371000.0) -> Tuple[float, float, float]:
        """Convert lat/lon/depth to 3D Cartesian coordinates."""
        lat_rad = np.radians(self.latitude)
        lon_rad = np.radians(self.longitude)
        
        # Surface position
        x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = earth_radius * np.sin(lat_rad)
        
        return (x, y, z)
    
    def get_bounding_box(self, earth_radius: float = 6371000.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get the bounding box of this drillhole."""
        x, y, z = self.to_cartesian(earth_radius)
        
        # Simple bounding box approximation
        half_size = max(self.depth, self.diameter * 10)  # Exaggeration factor
        
        min_corner = (x - half_size, y - half_size, z - half_size)
        max_corner = (x + half_size, y + half_size, z + half_size)
        
        return min_corner, max_corner


class VisualizationQuality(Enum):
    """Quality levels for rendering."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ULTRA = 4


class BaseVisualizationAlgorithm:
    """Base class for visualization algorithms."""
    
    def __init__(self, exaggeration_factor: float = 1000.0):
        """
        Initialize the visualization algorithm.
        
        Args:
            exaggeration_factor: How much to exaggerate the drillhole depth for visibility
        """
        self.exaggeration_factor = exaggeration_factor
        self.earth_radius = 6371000.0  # meters
        
    def prepare_data(self, drillholes: List[Drillhole]) -> None:
        """Prepare drillholes data for rendering."""
        raise NotImplementedError
        
    def get_visible_drillholes(self, 
                               camera_position: Tuple[float, float, float],
                               camera_direction: Tuple[float, float, float],
                               quality: VisualizationQuality) -> List[Drillhole]:
        """
        Get drillholes that should be rendered based on camera position.
        
        Args:
            camera_position: (x, y, z) position of the camera
            camera_direction: (x, y, z) direction vector of the camera
            quality: Rendering quality level
            
        Returns:
            List of drillholes to render
        """
        raise NotImplementedError
        
    def get_render_geometry(self, drillhole: Drillhole) -> Dict:
        """
        Get the geometry data for rendering a drillhole.
        
        Returns:
            Dictionary containing vertices, normals, and indices for the pipe geometry
        """
        # Create a simple cylindrical pipe
        segments = 8  # Number of segments around the cylinder
        
        x, y, z = drillhole.to_cartesian(self.earth_radius)
        
        # Direction vector from surface toward Earth center
        length = drillhole.depth * self.exaggeration_factor
        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)
        
        # Create cylinder vertices
        vertices = []
        normals = []
        
        # Create two circles (top and bottom of cylinder)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            
            # Find perpendicular vectors for the circle
            if abs(direction[2]) < 0.9:
                perp1 = np.cross(direction, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(direction, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Top circle
            offset = drillhole.diameter * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append([x + offset[0], y + offset[1], z + offset[2]])
            normals.append(offset / np.linalg.norm(offset))
            
            # Bottom circle
            bottom_pos = np.array([x, y, z]) - direction * length
            vertices.append([bottom_pos[0] + offset[0], 
                           bottom_pos[1] + offset[1], 
                           bottom_pos[2] + offset[2]])
            normals.append(offset / np.linalg.norm(offset))
        
        # Create indices for triangles
        indices = []
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # Side faces (two triangles per quad)
            indices.extend([
                2*i, 2*i+1, 2*next_i,
                2*next_i, 2*i+1, 2*next_i+1
            ])
        
        return {
            'vertices': np.array(vertices),
            'normals': np.array(normals),
            'indices': np.array(indices)
        }

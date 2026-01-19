
import logging
from typing import List, Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class TerrainType(Enum):
    MOUNTAIN = "mountain"
    FOREST = "forest"
    GRASSLAND = "grassland"
    URBAN = "urban"
    WATER = "water"
    UNKNOWN = "unknown"


class TerrainClassifier:
    
    def __init__(self):
        self.terrain_rules = {
            "mountain": {"min_elevation": 500, "slope_threshold": 10},
            "forest": {"vegetation_index": 0.7},
            "grassland": {"vegetation_index": 0.4},
            "urban": {"building_density": 0.5},
            "water": {"water_coverage": 0.8}
        }
        logger.info("TerrainClassifier Запуск")
    
    def classify_point(self, lat: float, lon: float, elevation: float = 0.0) -> str:
        if elevation > 500:
            return TerrainType.MOUNTAIN.value
        elif 40 < lat < 60:
            return TerrainType.FOREST.value
        else:
            return TerrainType.GRASSLAND.value
    
    def classify_track(self, points: List[Dict]) -> Dict:
        classifications = {}
        terrain_counts = {}
        
        for i, point in enumerate(points):
            lat = point.get("latitude", 0)
            lon = point.get("longitude", 0)
            elevation = point.get("altitude", 0)
            
            terrain = self.classify_point(lat, lon, elevation)
            classifications[i] = terrain
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
        
        dominant = max(terrain_counts, key=terrain_counts.get) if terrain_counts else "unknown"
        
        return {
            "classifications": classifications,
            "terrain_counts": terrain_counts,
            "dominant_terrain": dominant,
            "total_points": len(points)
        }
    
    def detect_terrain_transitions(self, points: List[Dict]) -> List[int]:
        transitions = []
        previous_terrain = None
        
        for i, point in enumerate(points):
            lat = point.get("latitude", 0)
            lon = point.get("longitude", 0)
            elevation = point.get("altitude", 0)
            
            current_terrain = self.classify_point(lat, lon, elevation)
            
            if previous_terrain is not None and current_terrain != previous_terrain:
                transitions.append(i)
            
            previous_terrain = current_terrain
        
        return transitions
    
    def get_terrain_properties(self, terrain_type: str) -> Dict:
        return self.terrain_rules.get(terrain_type, {})


TerrainAnalyzer = TerrainClassifier

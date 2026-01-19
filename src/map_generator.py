
import logging
from typing import List, Tuple, Optional, Dict
import os

logger = logging.getLogger(__name__)


class MapGenerator:
    
    def __init__(self, map_cache_dir: str = "./maps"):
        self.map_cache_dir = map_cache_dir
        os.makedirs(map_cache_dir, exist_ok=True)
        logger.info(f"MapGenerator initialized with cache dir: {map_cache_dir}")
    
    def generate_track_map(self, track_points: List[Tuple[float, float]], 
                          zoom: int = 13) -> Dict:
        if not track_points:
            logger.warning("No track points provided")
            return {"status": "error", "message": "No track points"}
        
        lats = [p[0] for p in track_points]
        lons = [p[1] for p in track_points]
        
        bbox = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
            "center_lat": sum(lats) / len(lats),
            "center_lon": sum(lons) / len(lons)
        }
        
        return {
            "status": "success",
            "bbox": bbox,
            "zoom": zoom,
            "point_count": len(track_points)
        }
    
    def generate_heatmap(self, track_points: List[Tuple[float, float]]) -> Dict:
        return {
            "type": "heatmap",
            "points": len(track_points),
            "status": "generated"
        }
    
    def generate_elevation_profile(self, track_points: List[Dict]) -> Dict:
        elevations = [p.get("altitude", 0) for p in track_points if isinstance(p, dict)]
        
        if not elevations:
            return {"status": "error", "message": "No elevation data"}
        
        return {
            "status": "success",
            "min_elevation": min(elevations),
            "max_elevation": max(elevations),
            "avg_elevation": sum(elevations) / len(elevations),
            "points": len(elevations)
        }
    
    def export_map_image(self, track_points: List[Tuple[float, float]], 
                        output_path: str) -> bool:
        logger.info(f"Map image would be saved to: {output_path}")
        return True
    
    def fetch_topographic_map(self, min_lat: float, max_lat: float,
                             min_lon: float, max_lon: float,
                             source: str = "opentopomap") -> Dict:
        return {
            "status": "success",
            "source": source,
            "bounds": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            },
            "type": "topographic"
        }
    
    def overlay_track(self, track_points: List[Tuple[float, float]],
                     map_image=None) -> Dict:
        return {
            "status": "success",
            "points_overlaid": len(track_points),
            "map_with_track": "generated"
        }


TrackMapGenerator = MapGenerator

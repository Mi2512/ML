
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class OSMLoader:
    
    def __init__(self):
        logger.info("OSMLoader Запуск")
        try:
            from external.osm_track_loader import OSMTrackLoader as ExistingLoader
            self.loader = ExistingLoader()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not import existing OSM loader: {e}")
            self.loader = None
    
    def load_routes_in_bbox(self, min_lat: float, max_lat: float, 
                           min_lon: float, max_lon: float, 
                           route_types: Optional[List[str]] = None) -> List[Dict]:
        if self.loader:
            return self.loader.load_routes_in_bbox(
                min_lat, max_lat, min_lon, max_lon, route_types
            )
        
        logger.warning("No OSM loader available, returning empty list")
        return []
    
    def load_gpx(self, gpx_url: str) -> Dict:
        if self.loader:
            return self.loader.load_gpx(gpx_url)
        
        logger.warning("No OSM loader available")
        return {"status": "error"}
    
    def load_amenities_in_bbox(self, min_lat: float, max_lat: float,
                              min_lon: float, max_lon: float,
                              amenity_types: Optional[List[str]] = None) -> List[Dict]:
        if self.loader:
            return self.loader.load_amenities_in_bbox(
                min_lat, max_lat, min_lon, max_lon, amenity_types
            )
        
        logger.warning("No OSM loader available")
        return []
    
    def load_elevation_data(self, lat: float, lon: float) -> Optional[float]:
        if self.loader and hasattr(self.loader, 'load_elevation_data'):
            return self.loader.load_elevation_data(lat, lon)
        
        logger.warning("Elevation loading not available")
        return None


OpenStreetMapLoader = OSMLoader

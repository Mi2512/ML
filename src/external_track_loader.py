
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ExternalTrackLoader:
    
    def __init__(self):
        logger.info("ExternalTrackLoader Запуск")
        try:
            from external.osm_track_loader import OSMTrackLoader
            self.osm_loader = OSMTrackLoader()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not import OSM loader: {e}")
            self.osm_loader = None
    
    def load_from_osm(self, min_lat: float, max_lat: float,
                     min_lon: float, max_lon: float,
                     route_types: Optional[List[str]] = None) -> List[Dict]:
        if self.osm_loader:
            return self.osm_loader.load_routes_in_bbox(
                min_lat, max_lat, min_lon, max_lon, route_types
            )
        
        return []
    
    def load_gpx_file(self, file_path: str) -> Dict:
        try:
            from track_parser import GPXParser
            parser = GPXParser()
            return parser.parse_gpx(file_path)
        except (ImportError, Exception) as e:
            logger.error(f"Failed to load GPX: {e}")
            return {"status": "error"}
    
    def load_tcx_file(self, file_path: str) -> Dict:
        try:
            from track_parser import TCXParser
            parser = TCXParser()
            return parser.parse_tcx(file_path)
        except (ImportError, Exception) as e:
            logger.error(f"Failed to load TCX: {e}")
            return {"status": "error"}
    
    def load_geojson_file(self, file_path: str) -> Dict:
        try:
            from track_parser import GeoJSONParser
            parser = GeoJSONParser()
            return parser.parse_geojson(file_path)
        except (ImportError, Exception) as e:
            logger.error(f"Failed to load GeoJSON: {e}")
            return {"status": "error"}
    
    def merge_tracks(self, tracks: List[Dict]) -> Dict:
        if not tracks:
            return {"status": "error", "message": "No tracks to merge"}
        
        merged = {
            "points": [],
            "source_count": len(tracks),
            "metadata": {}
        }
        
        for track in tracks:
            if "points" in track:
                merged["points"].extend(track["points"])
        
        return merged


ExternalSourceLoader = ExternalTrackLoader

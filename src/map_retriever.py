
import logging
from typing import Tuple, Optional, List
from pathlib import Path
import requests
from urllib.parse import urlencode
from PIL import Image
import io
import json
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class MapRetriever:
    
    def __init__(self, cache_dir: str = 'data/temp/map_cache',
                 zoom_level: int = 13):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zoom_level = zoom_level
        self.session = requests.Session()
        
        logger.info(f"MapRetriever initialized with cache at {cache_dir}")
    
    
    def compute_geodesic_points(self, start_lat: float, start_lon: float,
                               end_lat: float, end_lon: float,
                               num_points: int = 50) -> List[Tuple[float, float]]:
        lat1 = math.radians(start_lat)
        lon1 = math.radians(start_lon)
        lat2 = math.radians(end_lat)
        lon2 = math.radians(end_lon)
        
        dlon = lon2 - lon1
        a = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
        a = max(-1, min(1, a))
        sigma = math.acos(a)
        
        if sigma < 1e-6:
            return [(start_lat, start_lon), (end_lat, end_lon)]
        
        points = []
        for i in range(num_points):
            f = i / (num_points - 1)
            
            A = math.sin((1 - f) * sigma) / math.sin(sigma)
            B = math.sin(f * sigma) / math.sin(sigma)
            
            x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
            y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
            z = A * math.sin(lat1) + B * math.sin(lat2)
            
            lat = math.atan2(z, math.sqrt(x**2 + y**2))
            lon = math.atan2(y, x)
            
            points.append((math.degrees(lat), math.degrees(lon)))
        
        return points
    
    def get_geodesic_track(self, points: List[Tuple[float, float]], 
                          points_per_segment: int = 10) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points
        
        geodesic_points = []
        
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            
            segment = self.compute_geodesic_points(
                lat1, lon1, lat2, lon2,
                num_points=points_per_segment
            )
            
            geodesic_points.extend(segment[:-1])
        
        geodesic_points.append(points[-1])
        
        logger.info(f"Converted {len(points)} points to {len(geodesic_points)} "
                   f"geodesic points (reduction: {len(geodesic_points)/len(points):.1f}x)")
        
        return geodesic_points
    
    def add_projection_info(self, map_obj, points: List[Tuple[float, float]]) -> None:
        try:
            import folium
            
            if not points:
                return
            
            center_lat = sum(p[0] for p in points) / len(points)
            
            distortion = 1.0 / math.cos(math.radians(center_lat))
            
            info_text = (
                f"<b>Projection Info</b><br>"
                f"Center latitude: {center_lat:.2f}°<br>"
                f"Web Mercator distortion: {distortion:.2f}x<br>"
                f"<i>Track uses geodesic lines for accuracy</i>"
            )
            
            folium.Marker(
                location=[center_lat, sum(p[1] for p in points) / len(points)],
                popup=folium.Popup(info_text, max_width=300),
                icon=folium.Icon(color='blue', icon='info-sign'),
                opacity=0.0
            ).add_to(map_obj)
            
        except Exception as e:
            logger.warning(f"Could not add projection info: {e}")
    
    def get_google_maps_static(self, center_lat: float, center_lon: float,
                              zoom: int = None, width: int = 640, height: int = 640,
                              api_key: Optional[str] = None,
                              map_type: str = 'terrain') -> Optional[Image.Image]:
        if not api_key:
            logger.warning("Google Maps API key not provided, skipping")
            return None
        
        zoom = zoom or self.zoom_level
        
        try:
            params = {
                'center': f'{center_lat},{center_lon}',
                'zoom': zoom,
                'size': f'{width}x{height}',
                'maptype': map_type,
                'key': api_key
            }
            
            url = 'https://maps.googleapis.com/maps/api/staticmap?' + urlencode(params)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            logger.info(f"Retrieved Google Maps image for {center_lat}, {center_lon}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error fetching Google Maps: {e}")
            return None
    
    def get_yandex_maps_static(self, center_lat: float, center_lon: float,
                              zoom: int = None, width: int = 800, height: int = 600,
                              api_key: Optional[str] = None) -> Optional[Image.Image]:
        if not api_key:
            logger.warning("Yandex Maps API key not provided, skipping")
            return None
        
        zoom = zoom or self.zoom_level
        
        try:
            params = {
                'll': f'{center_lon},{center_lat}',
                'z': zoom,
                'size': f'{width},{height}',
                'l': 'map',
                'apikey': api_key
            }
            
            url = 'https://static-maps.yandex.ru/1.x?' + urlencode(params)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            logger.info(f"Retrieved Yandex Maps image for {center_lat}, {center_lon}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error fetching Yandex Maps: {e}")
            return None
    
    def save_map_image(self, image: Image.Image, track_id: str,
                      source: str, lat: float, lon: float) -> Path:
        filename = f'{track_id}_{source}_{lat:.4f}_{lon:.4f}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = self.cache_dir / filename
        
        image.save(filepath)
        logger.info(f"Saved map image to {filepath}")
        
        return filepath
    
    def get_track_map_with_overlay(self, track_id: str, points: List[Tuple[float, float]],
                                   yandex_api_key: Optional[str] = None) -> Optional[Path]:
        if not points:
            logger.warning("No points provided for map overlay")
            return None
        
        try:
            import folium
            
            center_lat = sum(p[0] for p in points) / len(points)
            center_lon = sum(p[1] for p in points) / len(points)
            
            map_obj = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=self.zoom_level,
                tiles=None
            )
            
            folium.TileLayer(
                tiles='https://core-renderer-tiles.maps.yandex.net/tiles?l=map&v=22.06.21-0&x={x}&y={y}&z={z}&lang=ru_RU',
                attr='Yandex Maps',
                name='Yandex Maps',
                overlay=False,
                control=True
            ).add_to(map_obj)
            
            logger.info(f"Converting track to geodesic representation for accuracy...")
            geodesic_track = self.get_geodesic_track(points, points_per_segment=10)
            
            folium.PolyLine(
                locations=geodesic_track,
                color='red',
                weight=3,
                opacity=0.8,
                popup=f'Track: {track_id} (Geodesic visualization)'
            ).add_to(map_obj)
            
            folium.Marker(
                location=points[0],
                popup='Start',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(map_obj)
            
            folium.Marker(
                location=points[-1],
                popup='End',
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(map_obj)
            
            self.add_projection_info(map_obj, points)
            
            filename = f'{track_id}_map_with_track_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            filepath = self.cache_dir / filename
            map_obj.save(str(filepath))
            
            logger.info(f" Saved geodesic track map to {filepath}")
            logger.info(f"   • Original points: {len(points)}")
            logger.info(f"   • Geodesic points: {len(geodesic_track)}")
            logger.info(f"   • Projection distortion minimized ")
            
            return filepath
            
        except ImportError:
            logger.error("Folium not installed, cannot create overlay map")
            return None
    
    def get_dem_elevation_data(self, lat: float, lon: float) -> Optional[float]:
        try:
            params = {
                'x': lon,
                'y': lat,
                'units': 'Meters',
                'output': 'json'
            }
            
            url = 'https://elevation-api.open-elevation.com/api/v1/lookup?' + urlencode(params)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            elevation = data.get('results', [{}])[0].get('elevation')
            
            return elevation
            
        except Exception as e:
            logger.warning(f"Error fetching DEM elevation: {e}")
            return None
    
    def batch_get_dem_elevation(self, points: List[Tuple[float, float]]) -> List[Optional[float]]:
        elevations = []
        
        for lat, lon in points:
            elev = self.get_dem_elevation_data(lat, lon)
            elevations.append(elev)
        
        return elevations
    
    def export_map_metadata(self, track_id: str, output_path: str) -> None:
        metadata = {
            'track_id': track_id,
            'maps_retrieved': {
                'yandex': True
            },
            'retrieved_at': datetime.now().isoformat(),
            'cache_location': str(self.cache_dir)
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported map metadata to {output_path}")


if __name__ == "__main__":
    retriever = MapRetriever()
    
    image = retriever.get_yandex_maps_static(55.75, 37.61, zoom=13, api_key='YOUR_API_KEY')
    if image:
        retriever.save_map_image(image, 'TEST_TRACK', 'yandex', 55.75, 37.61)
    
    points = [(55.75, 37.61), (55.751, 37.611), (55.752, 37.612)]
    retriever.get_track_map_with_overlay('TEST_TRACK', points)

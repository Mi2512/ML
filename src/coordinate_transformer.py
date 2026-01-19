
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import math
from pyproj import Transformer

logger = logging.getLogger(__name__)


@dataclass
class CoordinatePoint:
    latitude: float
    longitude: float
    altitude: float = 0.0
    crs: str = "EPSG:4326"


class CoordinateTransformer:
    
    EARTH_RADIUS_M = 6371000
    EARTH_RADIUS_KM = 6371
    WGS84_A = 6378137.0
    WGS84_B = 6356752.314245
    WGS84_E2 = 0.00669437999014132
    
    def __init__(self):
        self.wgs84_to_web_merc = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:3857",
            always_xy=True
        )
        self.web_merc_to_wgs84 = Transformer.from_crs(
            "EPSG:3857",
            "EPSG:4326",
            always_xy=True
        )
        
        logger.info("CoordinateTransformer Запуск with WGS84 and Web Mercator projections")
    
    def wgs84_to_web_mercator(self, lat: float, lon: float) -> Tuple[float, float]:
        x, y = self.wgs84_to_web_merc.transform(lon, lat)
        return x, y
    
    def web_mercator_to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        lon, lat = self.web_merc_to_wgs84.transform(x, y)
        return lat, lon
    
    def get_utm_zone(self, lon: float) -> int:
        return int((lon + 180) / 6) + 1
    
    def wgs84_to_utm(self, lat: float, lon: float) -> Tuple[float, float, int]:
        zone = self.get_utm_zone(lon)
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}",
            always_xy=True
        )
        x, y = transformer.transform(lon, lat)
        return x, y, zone
    
    def utm_to_wgs84(self, easting: float, northing: float, zone: int, 
                     is_northern: bool = True) -> Tuple[float, float]:
        epsg_code = 32600 + zone if is_northern else 32700 + zone
        transformer = Transformer.from_crs(
            f"EPSG:{epsg_code}",
            "EPSG:4326",
            always_xy=True
        )
        lon, lat = transformer.transform(easting, northing)
        return lat, lon
    
    def calculate_distance(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        return self.vincenty_distance(lat1, lon1, lat2, lon2)
    
    def transform_wgs84_to_web_mercator(self, lat: float, lon: float) -> Tuple[float, float]:
        return self.wgs84_to_web_mercator(lat, lon)
    
    def transform_wgs84_to_utm(self, lat: float, lon: float) -> Tuple[float, float, int]:
        return self.wgs84_to_utm(lat, lon)
    
    def haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return self.EARTH_RADIUS_M * c
    
    def vincenty_distance(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        U1 = math.atan((1 - 0.0818192 * 0.0818192) * math.tan(lat1_rad))
        U2 = math.atan((1 - 0.0818192 * 0.0818192) * math.tan(lat2_rad))
        
        sin_U1 = math.sin(U1)
        cos_U1 = math.cos(U1)
        sin_U2 = math.sin(U2)
        cos_U2 = math.cos(U2)
        
        lambda_val = dlon
        lambda_prev = float('inf')
        iteration = 0
        max_iterations = 100
        
        while abs(lambda_val - lambda_prev) > 1e-12 and iteration < max_iterations:
            iteration += 1
            lambda_prev = lambda_val
            
            sin_lambda = math.sin(lambda_val)
            cos_lambda = math.cos(lambda_val)
            
            sin_sigma = math.sqrt(
                (cos_U2 * sin_lambda)**2 +
                (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda)**2
            )
            
            if sin_sigma == 0:
                return 0.0
            
            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
            sigma = math.atan2(sin_sigma, cos_sigma)
            
            sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
            cos_sq_alpha = 1 - sin_alpha**2
            cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha if cos_sq_alpha != 0 else 0
            
            C = 0.0818192**2 / 16 * cos_sq_alpha * (4 + 0.0818192**2 * cos_sq_alpha)
            
            lambda_val = dlon + (1 - C) * 0.0818192 * sin_alpha * (
                sigma + C * sin_sigma * (
                    cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                )
            )
        
        u_sq = cos_sq_alpha * (self.WGS84_A**2 - self.WGS84_B**2) / (self.WGS84_B**2)
        A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
        
        delta_sigma = B * sin_sigma * (
            cos_2sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2) -
                B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos_2sigma_m**2)
            )
        )
        
        s = self.WGS84_B * A * (sigma - delta_sigma)
        
        return s
    
    def calculate_bearing(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing_deg = math.degrees(bearing)
        
        return (bearing_deg + 360) % 360
    
    def destination_point(self, lat: float, lon: float, bearing: float, 
                         distance_m: float) -> Tuple[float, float]:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        angular_distance = distance_m / self.EARTH_RADIUS_M
        
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_bearing = math.sin(bearing_rad)
        cos_bearing = math.cos(bearing_rad)
        sin_angular = math.sin(angular_distance)
        cos_angular = math.cos(angular_distance)
        
        lat2_rad = math.asin(
            sin_lat * cos_angular + cos_lat * sin_angular * cos_bearing
        )
        
        lon2_rad = lon_rad + math.atan2(
            sin_bearing * sin_angular * cos_lat,
            cos_angular - sin_lat * math.sin(lat2_rad)
        )
        
        return math.degrees(lat2_rad), math.degrees(lon2_rad)
    
    def distance_at_latitude(self, latitude: float) -> float:
        lat_rad = math.radians(latitude)
        return self.EARTH_RADIUS_M * math.cos(lat_rad) * (2 * math.pi / 360)
    
    def meters_per_degree_latitude(self) -> float:
        return self.EARTH_RADIUS_M * (2 * math.pi / 360)
    
    def validate_coordinates(self, lat: float, lon: float, 
                            strict: bool = False) -> bool:
        lat_valid = -90 <= lat <= 90
        lon_valid = -180 <= lon <= 180
        
        return lat_valid and lon_valid
    
    def batch_distance_calculation(self, points: List[Tuple[float, float]]) -> List[float]:
        distances = []
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            dist = self.vincenty_distance(lat1, lon1, lat2, lon2)
            distances.append(dist)
        
        return distances
    
    def calculate_track_statistics(self, points: List[Tuple[float, float]], 
                                   altitudes: Optional[List[float]] = None) -> Dict:
        if len(points) < 2:
            return {}
        
        distances = self.batch_distance_calculation(points)
        total_distance = sum(distances)
        
        stats = {
            'total_distance_m': total_distance,
            'total_distance_km': total_distance / 1000,
            'point_count': len(points),
            'segment_count': len(distances),
            'average_segment_distance_m': np.mean(distances) if distances else 0,
            'max_segment_distance_m': max(distances) if distances else 0,
            'min_segment_distance_m': min(distances) if distances else 0,
        }
        
        if altitudes and len(altitudes) == len(points):
            elevation_gains = []
            elevation_losses = []
            
            for i in range(len(altitudes) - 1):
                diff = altitudes[i+1] - altitudes[i]
                if diff > 0:
                    elevation_gains.append(diff)
                else:
                    elevation_losses.append(-diff)
            
            stats['total_elevation_gain_m'] = sum(elevation_gains)
            stats['total_elevation_loss_m'] = sum(elevation_losses)
            stats['total_elevation_change_m'] = stats['total_elevation_gain_m'] + stats['total_elevation_loss_m']
            stats['elevation_range_m'] = max(altitudes) - min(altitudes)
        
        return stats


if __name__ == "__main__":
    transformer = CoordinateTransformer()
    
    lat1, lon1 = 55.7558, 37.6173
    lat2, lon2 = 59.9311, 30.3609
    
    distance = transformer.vincenty_distance(lat1, lon1, lat2, lon2)
    print(f"Distance Moscow-SPb: {distance/1000:.2f} km")
    
    bearing = transformer.calculate_bearing(lat1, lon1, lat2, lon2)
    print(f"Bearing Moscow-SPb: {bearing:.2f}°")
    
    x, y, zone = transformer.wgs84_to_utm(lat1, lon1)
    print(f"Moscow in UTM: {x:.2f}, {y:.2f}, Zone {zone}")
    
    merc_x, merc_y = transformer.wgs84_to_web_mercator(lat1, lon1)
    print(f"Moscow in Web Mercator: {merc_x:.2f}, {merc_y:.2f}")


import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


WGS84_A = 6378137.0
WGS84_B = 6356752.314245
WGS84_F = 1 / 298.257223563
WGS84_E2 = 6.69437999014e-3

BOUNDS_MIN_LAT = 41.0
BOUNDS_MAX_LAT = 81.0
BOUNDS_MIN_LON = 19.0
BOUNDS_MAX_LON = 169.0

MAX_ROTATION_ANGLE = 45.0
MAX_AZIMUTH_SHIFT = 10000.0
MAX_ELEVATION_JITTER = 15.0
MAX_TEMPORAL_SHIFT_DAYS = 30
MAX_SLOPE_DEGREES = 89.0


@dataclass
class GeoPoint:
    lat: float
    lon: float
    elevation: float
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Invalid longitude: {self.lon}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lat': self.lat,
            'lon': self.lon,
            'elevation': self.elevation,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeoPoint':
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            lat=data['lat'],
            lon=data['lon'],
            elevation=data['elevation'],
            timestamp=timestamp
        )


@dataclass
class TransformationMatrix:
    transform_type: str
    parameters: Dict[str, Any]
    center_point: Optional[GeoPoint] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transform_type': self.transform_type,
            'parameters': self.parameters,
            'center_point': self.center_point.to_dict() if self.center_point else None,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformationMatrix':
        center = None
        if data.get('center_point'):
            center = GeoPoint.from_dict(data['center_point'])
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            transform_type=data['transform_type'],
            parameters=data['parameters'],
            center_point=center,
            timestamp=timestamp
        )


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return WGS84_A * c


def vincenty_direct(lat: float, lon: float, azimuth: float, distance: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    alpha1 = math.radians(azimuth)
    
    sin_alpha1 = math.sin(alpha1)
    cos_alpha1 = math.cos(alpha1)
    
    tan_U1 = (1 - WGS84_F) * math.tan(lat_rad)
    cos_U1 = 1 / math.sqrt(1 + tan_U1 ** 2)
    sin_U1 = tan_U1 * cos_U1
    
    sigma1 = math.atan2(tan_U1, cos_alpha1)
    sin_alpha = cos_U1 * sin_alpha1
    cos2_alpha = 1 - sin_alpha ** 2
    
    u2 = cos2_alpha * (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    
    sigma = distance / (WGS84_B * A)
    sigma_prev = 2 * math.pi
    
    while abs(sigma - sigma_prev) > 1e-12:
        cos_2sigma_m = math.cos(2 * sigma1 + sigma)
        sin_sigma = math.sin(sigma)
        cos_sigma = math.cos(sigma)
        
        delta_sigma = B * sin_sigma * (
            cos_2sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) -
                B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2)
            )
        )
        
        sigma_prev = sigma
        sigma = distance / (WGS84_B * A) + delta_sigma
    
    tmp = sin_U1 * sin_sigma - cos_U1 * cos_sigma * cos_alpha1
    lat2_rad = math.atan2(
        sin_U1 * cos_sigma + cos_U1 * sin_sigma * cos_alpha1,
        (1 - WGS84_F) * math.sqrt(sin_alpha ** 2 + tmp ** 2)
    )
    
    lambda_val = math.atan2(
        sin_sigma * sin_alpha1,
        cos_U1 * cos_sigma - sin_U1 * sin_sigma * cos_alpha1
    )
    
    C = WGS84_F / 16 * cos2_alpha * (4 + WGS84_F * (4 - 3 * cos2_alpha))
    L = lambda_val - (1 - C) * WGS84_F * sin_alpha * (
        sigma + C * sin_sigma * (
            cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
        )
    )
    
    lon2_rad = lon_rad + L
    
    return math.degrees(lat2_rad), math.degrees(lon2_rad)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360


def calculate_center_point(points: List[GeoPoint]) -> GeoPoint:
    if not points:
        raise ValueError("Cannot calculate center of empty point list")
    
    x_sum = y_sum = z_sum = 0
    for point in points:
        lat_rad = math.radians(point.lat)
        lon_rad = math.radians(point.lon)
        
        x_sum += math.cos(lat_rad) * math.cos(lon_rad)
        y_sum += math.cos(lat_rad) * math.sin(lon_rad)
        z_sum += math.sin(lat_rad)
    
    n = len(points)
    x_avg = x_sum / n
    y_avg = y_sum / n
    z_avg = z_sum / n
    
    lon_center = math.atan2(y_avg, x_avg)
    hyp = math.sqrt(x_avg ** 2 + y_avg ** 2)
    lat_center = math.atan2(z_avg, hyp)
    
    elevation_center = sum(p.elevation for p in points) / n
    
    return GeoPoint(
        lat=math.degrees(lat_center),
        lon=math.degrees(lon_center),
        elevation=elevation_center
    )


def is_within_bounds(lat: float, lon: float) -> bool:
    return (BOUNDS_MIN_LAT <= lat <= BOUNDS_MAX_LAT and
            BOUNDS_MIN_LON <= lon <= BOUNDS_MAX_LON)


def validate_track_bounds(points: List[GeoPoint]) -> Tuple[bool, List[str]]:
    errors = []
    
    for i, point in enumerate(points):
        if not is_within_bounds(point.lat, point.lon):
            errors.append(f"Point {i} out of bounds: ({point.lat:.4f}, {point.lon:.4f})")
    
    return len(errors) == 0, errors


def calculate_elevation_stats(points: List[GeoPoint]) -> Dict[str, float]:
    if len(points) < 2:
        return {'gain': 0.0, 'loss': 0.0, 'max_slope_degrees': 0.0}
    
    gain = 0.0
    loss = 0.0
    max_slope = 0.0
    
    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]
        
        elevation_diff = curr.elevation - prev.elevation
        if elevation_diff > 0:
            gain += elevation_diff
        else:
            loss += abs(elevation_diff)
        
        horizontal_dist = haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)
        if horizontal_dist > 0:
            slope_rad = math.atan(abs(elevation_diff) / horizontal_dist)
            slope_deg = math.degrees(slope_rad)
            max_slope = max(max_slope, slope_deg)
    
    return {
        'gain': gain,
        'loss': loss,
        'max_slope_degrees': max_slope
    }


class GeodeticAugmentation:
    
    def __init__(self):
        self.transformation_history: List[TransformationMatrix] = []
    
    def geodesic_rotation(
        self,
        points: List[GeoPoint],
        rotation_angle: float,
        center_point: Optional[GeoPoint] = None
    ) -> Tuple[List[GeoPoint], TransformationMatrix]:
        if not points:
            raise ValueError("Cannot rotate empty track")
        
        if abs(rotation_angle) > MAX_ROTATION_ANGLE:
            raise ValueError(f"Rotation angle {rotation_angle}° exceeds maximum {MAX_ROTATION_ANGLE}°")
        
        if center_point is None:
            center_point = calculate_center_point(points)
        
        rotated_points = []
        
        for point in points:
            distance = haversine_distance(
                center_point.lat, center_point.lon,
                point.lat, point.lon
            )
            
            bearing = calculate_bearing(
                center_point.lat, center_point.lon,
                point.lat, point.lon
            )
            
            new_bearing = (bearing + rotation_angle) % 360
            
            new_lat, new_lon = vincenty_direct(
                center_point.lat, center_point.lon,
                new_bearing, distance
            )
            
            if not is_within_bounds(new_lat, new_lon):
                raise ValueError(f"Rotation would place point outside bounds: ({new_lat:.4f}, {new_lon:.4f})")
            
            rotated_points.append(GeoPoint(
                lat=new_lat,
                lon=new_lon,
                elevation=point.elevation,
                timestamp=point.timestamp
            ))
        
        transform_matrix = TransformationMatrix(
            transform_type='geodesic_rotation',
            parameters={
                'rotation_angle': rotation_angle,
                'original_center_lat': center_point.lat,
                'original_center_lon': center_point.lon
            },
            center_point=center_point
        )
        
        self.transformation_history.append(transform_matrix)
        
        return rotated_points, transform_matrix
    
    def azimuth_shift(
        self,
        points: List[GeoPoint],
        shift_distance: float,
        azimuth: float
    ) -> Tuple[List[GeoPoint], TransformationMatrix]:
        if not points:
            raise ValueError("Cannot shift empty track")
        
        if abs(shift_distance) > MAX_AZIMUTH_SHIFT:
            raise ValueError(f"Shift distance {shift_distance}m exceeds maximum {MAX_AZIMUTH_SHIFT}m")
        
        shifted_points = []
        
        for point in points:
            new_lat, new_lon = vincenty_direct(
                point.lat, point.lon,
                azimuth, shift_distance
            )
            
            if not is_within_bounds(new_lat, new_lon):
                raise ValueError(f"Shift would place point outside bounds: ({new_lat:.4f}, {new_lon:.4f})")
            
            shifted_points.append(GeoPoint(
                lat=new_lat,
                lon=new_lon,
                elevation=point.elevation,
                timestamp=point.timestamp
            ))
        
        transform_matrix = TransformationMatrix(
            transform_type='azimuth_shift',
            parameters={
                'shift_distance': shift_distance,
                'azimuth': azimuth
            }
        )
        
        self.transformation_history.append(transform_matrix)
        
        return shifted_points, transform_matrix
    
    def elevation_jitter(
        self,
        points: List[GeoPoint],
        jitter_magnitude: float = 10.0,
        preserve_gain_loss: bool = True,
        tolerance: float = 0.1
    ) -> Tuple[List[GeoPoint], TransformationMatrix]:
        if not points:
            raise ValueError("Cannot jitter empty track")
        
        if jitter_magnitude > MAX_ELEVATION_JITTER:
            raise ValueError(f"Jitter magnitude {jitter_magnitude}m exceeds maximum {MAX_ELEVATION_JITTER}m")
        
        original_stats = calculate_elevation_stats(points)
        
        np.random.seed(int(datetime.now().timestamp() * 1000) % (2**32))
        jitter_values = np.random.uniform(-jitter_magnitude, jitter_magnitude, len(points))
        
        jitter_smoothed = np.convolve(jitter_values, np.ones(3) / 3, mode='same')
        
        jittered_points = []
        for i, point in enumerate(points):
            new_elevation = point.elevation + jitter_smoothed[i]
            
            jittered_points.append(GeoPoint(
                lat=point.lat,
                lon=point.lon,
                elevation=new_elevation,
                timestamp=point.timestamp
            ))
        
        new_stats = calculate_elevation_stats(jittered_points)
        if new_stats['max_slope_degrees'] > MAX_SLOPE_DEGREES:
            raise ValueError(f"Jitter creates slope {new_stats['max_slope_degrees']:.1f}° exceeding {MAX_SLOPE_DEGREES}°")
        
        if preserve_gain_loss and len(points) > 1:
            gain_ratio = new_stats['gain'] / max(original_stats['gain'], 1.0)
            loss_ratio = new_stats['loss'] / max(original_stats['loss'], 1.0)
            
            if not (1 - tolerance <= gain_ratio <= 1 + tolerance) or \
               not (1 - tolerance <= loss_ratio <= 1 + tolerance):
                scale_factor = (original_stats['gain'] + original_stats['loss']) / \
                              (new_stats['gain'] + new_stats['loss'])
                
                base_elevation = points[0].elevation
                for i, point in enumerate(jittered_points):
                    elevation_diff = point.elevation - base_elevation
                    jittered_points[i].elevation = base_elevation + elevation_diff * scale_factor
        
        transform_matrix = TransformationMatrix(
            transform_type='elevation_jitter',
            parameters={
                'jitter_magnitude': jitter_magnitude,
                'preserve_gain_loss': preserve_gain_loss,
                'tolerance': tolerance,
                'jitter_values': jitter_smoothed.tolist()
            }
        )
        
        self.transformation_history.append(transform_matrix)
        
        return jittered_points, transform_matrix
    
    def temporal_shift(
        self,
        points: List[GeoPoint],
        shift_days: int
    ) -> Tuple[List[GeoPoint], TransformationMatrix]:
        if not points:
            raise ValueError("Cannot shift empty track")
        
        if abs(shift_days) > MAX_TEMPORAL_SHIFT_DAYS:
            raise ValueError(f"Temporal shift {shift_days} days exceeds maximum {MAX_TEMPORAL_SHIFT_DAYS} days")
        
        if not all(p.timestamp for p in points):
            raise ValueError("All points must have timestamps for temporal shift")
        
        shift_delta = timedelta(days=shift_days)
        shifted_points = []
        
        for point in points:
            new_timestamp = point.timestamp + shift_delta
            
            shifted_points.append(GeoPoint(
                lat=point.lat,
                lon=point.lon,
                elevation=point.elevation,
                timestamp=new_timestamp
            ))
        
        transform_matrix = TransformationMatrix(
            transform_type='temporal_shift',
            parameters={
                'shift_days': shift_days
            }
        )
        
        self.transformation_history.append(transform_matrix)
        
        return shifted_points, transform_matrix
    
    def get_transformation_history(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.transformation_history]
    
    def clear_history(self):
        self.transformation_history.clear()


if __name__ == '__main__':
    print("Geodetic Augmentation Module - Test")
    print("=" * 70)
    
    sample_points = [
        GeoPoint(55.7558, 120.4356, 1200.0, datetime(2026, 1, 15, 10, 0)),
        GeoPoint(55.7560, 120.4360, 1205.0, datetime(2026, 1, 15, 10, 5)),
        GeoPoint(55.7565, 120.4370, 1210.0, datetime(2026, 1, 15, 10, 10)),
        GeoPoint(55.7570, 120.4380, 1215.0, datetime(2026, 1, 15, 10, 15)),
    ]
    
    print(f"Original track: {len(sample_points)} points")
    print(f"Center: ({sample_points[0].lat:.4f}, {sample_points[0].lon:.4f})")
    
    augmenter = GeodeticAugmentation()
    
    print("\nTest 1: Geodesic Rotation (15°)")
    rotated, matrix1 = augmenter.geodesic_rotation(sample_points, 15.0)
    print(f" Rotated {len(rotated)} points")
    print(f"   First point: ({rotated[0].lat:.6f}, {rotated[0].lon:.6f})")
    
    print("\nTest 2: Azimuth Shift (1000m @ 90°)")
    shifted, matrix2 = augmenter.azimuth_shift(sample_points, 1000.0, 90.0)
    print(f" Shifted {len(shifted)} points")
    print(f"   First point: ({shifted[0].lat:.6f}, {shifted[0].lon:.6f})")
    
    print("\nTest 3: Elevation Jitter (±10m)")
    jittered, matrix3 = augmenter.elevation_jitter(sample_points, 10.0)
    print(f" Jittered {len(jittered)} points")
    print(f"   First elevation: {jittered[0].elevation:.2f}m (original: {sample_points[0].elevation:.2f}m)")
    
    print("\nTest 4: Temporal Shift (+7 days)")
    temporal, matrix4 = augmenter.temporal_shift(sample_points, 7)
    print(f" Shifted {len(temporal)} points")
    print(f"   First timestamp: {temporal[0].timestamp}")
    
    print("\n" + "=" * 70)
    print(f" All transformations completed: {len(augmenter.transformation_history)} in history")

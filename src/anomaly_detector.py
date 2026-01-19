
from .interpolator import GPSInterpolator
import logging

logger = logging.getLogger(__name__)


class GPSAnomalyDetector:
    
    def __init__(self):
        self.interpolator = GPSInterpolator()
    
    def detect_anomalies(self, points):
        coords = [(p.get('latitude'), p.get('longitude')) for p in points]
        timestamps = [p.get('timestamp') for p in points if p.get('timestamp')]
        
        result = self.interpolator.detect_and_fix_anomalies(
            track_id='temp',
            points=coords,
            timestamps=timestamps if timestamps else None
        )
        
        return result.get('anomalies', [])
    
    def detect_gps_anomalies(self, track_data):
        return self.detect_anomalies(track_data)
    
    def validate_coordinates(self, lat, lon):
        if not (-90 <= lat <= 90):
            return False
        if not (-180 <= lon <= 180):
            return False
        return True
    
    def detect_speed_violations(self, points, max_speed_kmh=50.0):
        violations = []
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            lat1, lon1 = p1.get('latitude'), p1.get('longitude')
            lat2, lon2 = p2.get('latitude'), p2.get('longitude')
            
            if None in [lat1, lon1, lat2, lon2]:
                continue
            
            import numpy as np
            R = 6371000
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            dlat_rad = np.radians(lat2 - lat1)
            dlon_rad = np.radians(lon2 - lon1)
            
            a = np.sin(dlat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_m = R * c
            
            t1 = p1.get('timestamp')
            t2 = p2.get('timestamp')
            
            if t1 and t2:
                from datetime import datetime
                try:
                    if isinstance(t1, str):
                        t1 = datetime.fromisoformat(str(t1))
                    if isinstance(t2, str):
                        t2 = datetime.fromisoformat(str(t2))
                    
                    time_delta = (t2 - t1).total_seconds()
                    
                    if time_delta > 0:
                        speed_ms = distance_m / time_delta
                        speed_kmh = speed_ms * 3.6
                        
                        if speed_kmh > max_speed_kmh:
                            violations.append(i)
                except:
                    pass
        
        return violations
    
    def correct_speed_anomalies(self, points, max_speed_kmh=50.0):
        violations = self.detect_speed_violations(points, max_speed_kmh)
        
        if not violations:
            return points
        
        coords = [(p.get('latitude'), p.get('longitude')) for p in points]
        
        corrected_coords = self.interpolator.linear_interpolation(coords, violations)
        
        corrected_points = []
        for i, point in enumerate(points):
            new_point = point.copy()
            if i < len(corrected_coords):
                new_point['latitude'] = corrected_coords[i][0]
                new_point['longitude'] = corrected_coords[i][1]
            corrected_points.append(new_point)
        
        return corrected_points
    
    def detect_duplicate_points(self, points, min_distance_m=1.0):
        import numpy as np
        duplicates = []
        
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            
            lat1, lon1 = p1.get('latitude'), p1.get('longitude')
            lat2, lon2 = p2.get('latitude'), p2.get('longitude')
            
            if None in [lat1, lon1, lat2, lon2]:
                continue
            
            R = 6371000
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            dlat_rad = np.radians(lat2 - lat1)
            dlon_rad = np.radians(lon2 - lon1)
            
            a = np.sin(dlat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            if distance < min_distance_m:
                duplicates.append(i)
        
        return duplicates
    
    def interpolate_missing_points(self, points, gap_indices=None):
        if not points or len(points) < 2:
            return points
        
        coords = [(p.get('latitude'), p.get('longitude')) for p in points]
        
        if gap_indices is None:
            gap_indices = [i for i, p in enumerate(points) 
                          if p.get('latitude') is None or p.get('longitude') is None]
        
        if not gap_indices:
            return points
        
        corrected_coords = self.interpolator.linear_interpolation(coords, gap_indices)
        
        result = []
        for i, point in enumerate(points):
            new_point = point.copy()
            if i < len(corrected_coords):
                new_point['latitude'] = corrected_coords[i][0]
                new_point['longitude'] = corrected_coords[i][1]
            result.append(new_point)
        
        return result
        
        return result


AnomalyDetector = GPSAnomalyDetector

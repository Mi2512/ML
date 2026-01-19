
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRecord:
    track_id: str
    point_index: int
    anomaly_type: str
    original_lat: Optional[float]
    original_lon: Optional[float]
    corrected_lat: Optional[float]
    corrected_lon: Optional[float]
    method: str
    confidence: float
    timestamp: str
    calculated_speed: Optional[float] = None
    distance_to_prev: Optional[float] = None
    distance_to_next: Optional[float] = None
    time_delta: Optional[float] = None
    speed_violation: bool = False


class GPSInterpolator:
    
    def __init__(self):
        self.anomaly_records: List[AnomalyRecord] = []
        logger.info("GPSInterpolator Запуск")
    
    def linear_interpolation(self, points: List[Tuple[float, float]], 
                            indices_to_fix: List[int]) -> List[Tuple[float, float]]:
        if not indices_to_fix:
            return points
        
        corrected_points = [p for p in points]
        
        for idx in sorted(indices_to_fix):
            if idx == 0 or idx == len(points) - 1:
                continue
            
            before_idx = idx - 1
            while before_idx >= 0 and before_idx in indices_to_fix:
                before_idx -= 1
            
            after_idx = idx + 1
            while after_idx < len(points) and after_idx in indices_to_fix:
                after_idx += 1
            
            if before_idx >= 0 and after_idx < len(points):
                lat_before, lon_before = corrected_points[before_idx]
                lat_after, lon_after = corrected_points[after_idx]
                
                num_segments = after_idx - before_idx
                progress = (idx - before_idx) / num_segments
                
                lat_interp = lat_before + (lat_after - lat_before) * progress
                lon_interp = lon_before + (lon_after - lon_before) * progress
                
                corrected_points[idx] = (lat_interp, lon_interp)
        
        return corrected_points
    
    def spline_interpolation(self, points: List[Tuple[float, float]], 
                            indices_to_fix: List[int], 
                            kind: str = 'cubic') -> List[Tuple[float, float]]:
        if not indices_to_fix or len(points) < 3:
            return self.linear_interpolation(points, indices_to_fix)
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        indices = list(range(len(points)))
        
        valid_idx = [i for i in indices if i not in indices_to_fix]
        valid_lats = [lats[i] for i in valid_idx]
        valid_lons = [lons[i] for i in valid_idx]
        
        if len(valid_idx) < 2:
            return self.linear_interpolation(points, indices_to_fix)
        
        try:
            f_lat = interp1d(valid_idx, valid_lats, kind=kind, fill_value='extrapolate')
            f_lon = interp1d(valid_idx, valid_lons, kind=kind, fill_value='extrapolate')
            
            corrected_points = []
            for i, (lat, lon) in enumerate(points):
                if i in indices_to_fix:
                    corrected_points.append((float(f_lat(i)), float(f_lon(i))))
                else:
                    corrected_points.append((lat, lon))
            
            return corrected_points
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, falling back to linear")
            return self.linear_interpolation(points, indices_to_fix)
    
    def kalman_filter(self, points: List[Tuple[float, float]], 
                     process_noise: float = 1e-5,
                     measurement_noise: float = 1e-4) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points
        
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        
        filtered_lats = self._kalman_filter_1d(lats, process_noise, measurement_noise)
        filtered_lons = self._kalman_filter_1d(lons, process_noise, measurement_noise)
        
        return list(zip(filtered_lats, filtered_lons))
    
    @staticmethod
    def _kalman_filter_1d(measurements: np.ndarray, 
                         process_noise: float,
                         measurement_noise: float) -> np.ndarray:
        n = len(measurements)
        x = np.zeros(n)
        P = np.ones(n)
        
        x[0] = measurements[0]
        P[0] = measurement_noise
        
        for i in range(1, n):
            x_pred = x[i-1]
            P_pred = P[i-1] + process_noise
            
            K = P_pred / (P_pred + measurement_noise)
            x[i] = x_pred + K * (measurements[i] - x_pred)
            P[i] = (1 - K) * P_pred
        
        return x
    
    def savitzky_golay_smoothing(self, points: List[Tuple[float, float]], 
                                window_length: int = 5,
                                polyorder: int = 2) -> List[Tuple[float, float]]:
        if len(points) < window_length:
            return points
        
        from scipy.signal import savgol_filter
        
        if window_length % 2 == 0:
            window_length += 1
        
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        
        try:
            smoothed_lats = savgol_filter(lats, window_length, polyorder)
            smoothed_lons = savgol_filter(lons, window_length, polyorder)
            
            return list(zip(smoothed_lats, smoothed_lons))
        except Exception as e:
            logger.warning(f"Savitzky-Golay smoothing failed: {e}")
            return points
    
    def gaussian_smoothing(self, points: List[Tuple[float, float]], 
                          sigma: float = 1.0) -> List[Tuple[float, float]]:
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        
        smoothed_lats = gaussian_filter1d(lats, sigma=sigma)
        smoothed_lons = gaussian_filter1d(lons, sigma=sigma)
        
        return list(zip(smoothed_lats, smoothed_lons))
    
    def detect_and_fix_anomalies(self, track_id: str,
                                points: List[Tuple[float, float]],
                                timestamps: List[str] = None,
                                speed_threshold_kmh: float = 50,
                                distance_threshold_m: float = 100,
                                from_transformer = None) -> Dict:
        if len(points) < 2:
            return {'anomalies_found': 0, 'corrections_applied': 0, 'speed_violations': 0}
        
        anomalies_found = []
        indices_to_fix = []
        speed_violations = []
        corrected_points = points.copy()
        
        point_times = []
        if timestamps:
            from datetime import datetime as dt
            for ts_str in timestamps:
                try:
                    point_times.append(dt.fromisoformat(ts_str))
                except:
                    point_times.append(None)
        
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            
            dlat = abs(lat2 - lat1)
            dlon = abs(lon2 - lon1)
            approx_dist_deg = np.sqrt(dlat**2 + dlon**2)
            
            distance_to_next = None
            time_delta = None
            calculated_speed = None
            speed_violation = False
            
            if approx_dist_deg < 0.0001:
                anomalies_found.append({
                    'type': 'gps_noise',
                    'index': i,
                    'description': f'Points {i} and {i+1} very close ({approx_dist_deg:.6f}°)'
                })
                indices_to_fix.append(i)
            
            if len(point_times) > i + 1 and point_times[i] and point_times[i + 1]:
                try:
                    from datetime import datetime as dt
                    R = 6371000
                    lat1_rad = np.radians(lat1)
                    lat2_rad = np.radians(lat2)
                    dlat_rad = np.radians(lat2 - lat1)
                    dlon_rad = np.radians(lon2 - lon1)
                    
                    a = np.sin(dlat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance_to_next = R * c
                    
                    time_delta = (point_times[i + 1] - point_times[i]).total_seconds()
                    
                    if time_delta > 0:
                        calculated_speed = distance_to_next / time_delta
                        speed_kmh = calculated_speed * 3.6
                        
                        if speed_kmh > speed_threshold_kmh:
                            speed_violation = True
                            speed_violations.append({
                                'index': i,
                                'calculated_speed_kmh': round(speed_kmh, 2),
                                'distance_m': round(distance_to_next, 2),
                                'time_s': round(time_delta, 2),
                                'description': f'Speed violation: {speed_kmh:.2f} km/h (threshold: {speed_threshold_kmh} km/h)'
                            })
                except Exception as e:
                    logger.warning(f"Could not calculate speed for point {i}: {str(e)}")
            
            if i > 0:
                lat_prev, lon_prev = points[i - 1]
                dlat_prev = abs(lat1 - lat_prev)
                dlon_prev = abs(lon1 - lon_prev)
                
                try:
                    R = 6371000
                    lat_prev_rad = np.radians(lat_prev)
                    lat1_rad = np.radians(lat1)
                    dlat_prev_rad = np.radians(lat1 - lat_prev)
                    dlon_prev_rad = np.radians(lon1 - lon_prev)
                    
                    a = np.sin(dlat_prev_rad/2)**2 + np.cos(lat_prev_rad) * np.cos(lat1_rad) * np.sin(dlon_prev_rad/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance_to_prev = R * c
                except:
                    distance_to_prev = None
            else:
                distance_to_prev = None
        
        if indices_to_fix:
            corrected_points = self.spline_interpolation(points, indices_to_fix, kind='cubic')
        
        for idx in indices_to_fix:
            record = AnomalyRecord(
                track_id=track_id,
                point_index=idx,
                anomaly_type='gps_noise',
                original_lat=points[idx][0],
                original_lon=points[idx][1],
                corrected_lat=corrected_points[idx][0],
                corrected_lon=corrected_points[idx][1],
                method='cubic_spline',
                confidence=0.85,
                timestamp=datetime.now().isoformat(),
                calculated_speed=None,
                distance_to_prev=distance_to_prev if idx > 0 else None,
                distance_to_next=None,
                time_delta=None,
                speed_violation=False
            )
            self.anomaly_records.append(record)
        
        for violation in speed_violations:
            idx = violation['index']
            logger.warning(f"Speed violation at point {idx}: {violation['description']}")
        
        logger.info(f"Track {track_id}: Detected {len(anomalies_found)} GPS anomalies, {len(speed_violations)} speed violations")
        
        return {
            'anomalies_found': len(anomalies_found),
            'corrections_applied': len(indices_to_fix),
            'speed_violations': len(speed_violations),
            'corrected_points': corrected_points,
            'anomaly_details': anomalies_found,
            'speed_violations_detail': speed_violations
        }
    
    def get_anomaly_records(self) -> List[Dict]:
        return [
            {
                'track_id': r.track_id,
                'point_index': r.point_index,
                'anomaly_type': r.anomaly_type,
                'original_coordinates': (r.original_lat, r.original_lon),
                'corrected_coordinates': (r.corrected_lat, r.corrected_lon),
                'method': r.method,
                'confidence': r.confidence,
                'timestamp': r.timestamp
            }
            for r in self.anomaly_records
        ]
    
    def export_corrections(self, output_path: str) -> None:
        import json
        
        records = self.get_anomaly_records()
        
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
        
        logger.info(f"Exported {len(records)} anomaly records to {output_path}")


if __name__ == "__main__":
    interpolator = GPSInterpolator()
    
    points = [
        (55.75, 37.61),
        (55.7505, 37.6105),
        (55.751, 37.611),
        (55.752, 37.612),
        (55.753, 37.613),
    ]
    
    print("Original points:", points)
    
    filtered = interpolator.kalman_filter(points)
    print("Kalman filtered:", filtered)
    
    result = interpolator.detect_and_fix_anomalies('TEST_TRACK', points)
    print("Correction result:", result)

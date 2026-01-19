
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RouteDataLoader:
    
    LATITUDE_MIN, LATITUDE_MAX = 45.0, 72.0
    LONGITUDE_MIN, LONGITUDE_MAX = 15.0, 180.0
    ALTITUDE_MIN, ALTITUDE_MAX = -100, 6000
    
    MAX_SPEED_KMH = 50
    MAX_DAILY_DISTANCE = 100
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.raw_data = None
        self.cleaned_data = None
        self.metadata = {
            'source': str(self.csv_path),
            'loaded_at': None,
            'total_tracks': 0,
            'total_points': 0,
            'validation_results': {},
            'anomalies_detected': {},
            'corrections_applied': {},
            'missing_values': {}
        }
        logger.info(f"RouteDataLoader initialized with path: {csv_path}")
    
    def load_raw_data(self, sample: Optional[int] = None) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"File not found: {self.csv_path}")
        
        logger.info(f"Loading data from {self.csv_path}")
        
        try:
            if sample:
                df = pd.read_csv(self.csv_path, nrows=sample)
                logger.info(f"Loaded sample of {sample} rows")
            else:
                df = pd.read_csv(self.csv_path)
            
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            
            logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            self.raw_data = df.copy()
            self.metadata['loaded_at'] = datetime.now().isoformat()
            self.metadata['total_points'] = len(df)
            self.metadata['total_tracks'] = df['track_id'].nunique()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Failed to parse CSV: {str(e)}")
    
    def validate_geodetic_data(self) -> Dict[str, any]:
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data
        validation_results = {
            'invalid_latitude': [],
            'invalid_longitude': [],
            'invalid_altitude': [],
            'missing_coordinates': [],
            'summary': {}
        }
        
        logger.info("Validating geodetic data")
        
        invalid_lat = df[
            (df['latitude'] < self.LATITUDE_MIN) | 
            (df['latitude'] > self.LATITUDE_MAX)
        ]
        if len(invalid_lat) > 0:
            validation_results['invalid_latitude'] = invalid_lat.index.tolist()
            logger.warning(f"Found {len(invalid_lat)} points with invalid latitude")
        
        invalid_lon = df[
            (df['longitude'] < self.LONGITUDE_MIN) | 
            (df['longitude'] > self.LONGITUDE_MAX)
        ]
        if len(invalid_lon) > 0:
            validation_results['invalid_longitude'] = invalid_lon.index.tolist()
            logger.warning(f"Found {len(invalid_lon)} points with invalid longitude")
        
        invalid_alt = df[
            (df['altitude'] < self.ALTITUDE_MIN) | 
            (df['altitude'] > self.ALTITUDE_MAX)
        ]
        if len(invalid_alt) > 0:
            validation_results['invalid_altitude'] = invalid_alt.index.tolist()
            logger.warning(f"Found {len(invalid_alt)} points with invalid altitude")
        
        missing_coords = df[df[['latitude', 'longitude']].isna().any(axis=1)]
        if len(missing_coords) > 0:
            validation_results['missing_coordinates'] = missing_coords.index.tolist()
            logger.warning(f"Found {len(missing_coords)} points with missing coordinates")
        
        validation_results['summary'] = {
            'total_points': len(df),
            'valid_points': len(df) - len(invalid_lat) - len(invalid_lon) - len(invalid_alt),
            'validity_percentage': (1 - (len(invalid_lat) + len(invalid_lon) + len(invalid_alt)) / len(df)) * 100
        }
        
        logger.info(f"Geodetic validation complete: {validation_results['summary']['validity_percentage']:.2f}% points valid")
        
        self.metadata['validation_results'] = validation_results
        return validation_results
    
    def detect_gps_anomalies(self) -> Dict[str, any]:
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data.copy()
        anomalies = {
            'excessive_speed': [],
            'gps_noise': [],
            'impossible_distance': [],
            'summary': {}
        }
        
        logger.info("Detecting GPS anomalies")
        
        df_sorted = df.sort_values(['track_id', 'point_index']).reset_index(drop=True)
        
        for track_id in df_sorted['track_id'].unique():
            track_df = df_sorted[df_sorted['track_id'] == track_id].copy()
            
            if len(track_df) < 2:
                continue
            
            track_df = track_df.reset_index(drop=True)
            
            for i in range(len(track_df) - 1):
                lat1, lon1 = track_df.loc[i, ['latitude', 'longitude']]
                lat2, lon2 = track_df.loc[i+1, ['latitude', 'longitude']]
                date1 = track_df.loc[i, 'date']
                date2 = track_df.loc[i+1, 'date']
                
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                R_km = 6371
                distance_km = R_km * c
                
                time_diff = (date2 - date1).total_seconds() / 3600
                
                if distance_km < 0.01 and time_diff > 0:
                    anomalies['gps_noise'].append({
                        'track_id': track_id,
                        'point_index': i,
                        'distance_m': distance_km * 1000,
                        'time_hours': time_diff
                    })
                
                if time_diff > 0:
                    speed_kmh = distance_km / time_diff
                    if speed_kmh > self.MAX_SPEED_KMH:
                        anomalies['excessive_speed'].append({
                            'track_id': track_id,
                            'point_index': i,
                            'speed_kmh': speed_kmh,
                            'distance_km': distance_km,
                            'time_hours': time_diff
                        })
        
        anomalies['summary'] = {
            'total_anomalies': len(anomalies['excessive_speed']) + len(anomalies['gps_noise']),
            'excessive_speed_count': len(anomalies['excessive_speed']),
            'gps_noise_count': len(anomalies['gps_noise'])
        }
        
        logger.info(f"GPS anomaly detection complete: {anomalies['summary']['total_anomalies']} anomalies found")
        
        self.metadata['anomalies_detected'] = anomalies
        return anomalies
    
    def analyze_missing_values(self) -> Dict[str, any]:
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data
        missing_analysis = {}
        
        logger.info("Analyzing missing values")
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_analysis[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'data_type': str(df[col].dtype),
                'recommendation': self._recommend_missing_strategy(col, missing_pct)
            }
        
        self.metadata['missing_values'] = missing_analysis
        
        for col, info in missing_analysis.items():
            if info['missing_count'] > 0:
                logger.info(f"  {col}: {info['missing_count']} missing ({info['missing_percentage']:.2f}%)")
        
        return missing_analysis
    
    @staticmethod
    def _recommend_missing_strategy(column: str, missing_pct: float) -> str:
        if missing_pct == 0:
            return "No action needed"
        elif missing_pct < 5:
            return "Can interpolate or use mean/median"
        elif missing_pct < 20:
            return "Interpolate with caution or use DEM for altitude"
        else:
            return "Consider dropping or flagging as unreliable"
    
    def detect_duplicates(self) -> Dict[str, any]:
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data
        duplicate_info = {
            'exact_duplicates': [],
            'coordinate_duplicates': [],
            'summary': {}
        }
        
        logger.info("Detecting duplicates")
        
        exact_dups = df.duplicated(keep=False)
        if exact_dups.any():
            duplicate_info['exact_duplicates'] = df[exact_dups].index.tolist()
            logger.warning(f"Found {len(duplicate_info['exact_duplicates'])} exact duplicate rows")
        
        for track_id in df['track_id'].unique():
            track_df = df[df['track_id'] == track_id]
            coord_dups = track_df.duplicated(subset=['latitude', 'longitude'], keep=False)
            if coord_dups.any():
                duplicate_info['coordinate_duplicates'].extend(
                    track_df[coord_dups].index.tolist()
                )
        
        duplicate_info['summary'] = {
            'exact_duplicate_count': len(duplicate_info['exact_duplicates']),
            'coordinate_duplicate_count': len(duplicate_info['coordinate_duplicates'])
        }
        
        logger.info(f"Duplicate detection complete: {duplicate_info['summary']['coordinate_duplicate_count']} coordinate duplicates found")
        
        return duplicate_info
    
    def get_basic_statistics(self) -> Dict[str, any]:
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data
        stats = {}
        
        logger.info("Computing basic statistics")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q1': float(df[col].quantile(0.25)),
                'q3': float(df[col].quantile(0.75))
            }
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_common': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
                'most_common_count': int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0
            }
        
        return stats
    
    def validate_all(self) -> bool:
        logger.info("=" * 60)
        logger.info("Running comprehensive Data validation")
        logger.info("=" * 60)
        
        try:
            self.validate_geodetic_data()
            self.detect_gps_anomalies()
            self.analyze_missing_values()
            self.detect_duplicates()
            
            logger.info("=" * 60)
            logger.info("VALIDATION Готово")
            logger.info("=" * 60)
            
            return True
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False
    
    def export_metadata(self, output_path: str) -> None:
        import json
        
        metadata_export = self.metadata.copy()
        metadata_export['exported_at'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(metadata_export, f, indent=2, default=str)
        
        logger.info(f"Metadata exported to {output_path}")
    
    def get_metadata(self) -> Dict:
        return self.metadata.copy()


def load_and_validate_routes(csv_path: str, sample: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    loader = RouteDataLoader(csv_path)
    loader.load_raw_data(sample=sample)
    loader.validate_all()
    
    return loader.raw_data, loader.get_metadata()


if __name__ == "__main__":
    csv_file = "dataset/large_route_dataset_20260117_143350.csv"
    
    loader = RouteDataLoader(csv_file)
    loader.load_raw_data()
    loader.validate_all()
    
    stats = loader.get_basic_statistics()
    print("\nBasic Statistics:")
    for col, stat in stats.items():
        print(f"  {col}: {stat}")


import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math

logger = logging.getLogger(__name__)



class Season(str, Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"


class TemperatureClass(str, Enum):
    VERY_COLD = "very_cold"
    COLD = "cold"
    COOL = "cool"
    WARM = "warm"
    HOT = "hot"


class AltitudeClass(str, Enum):
    LOWLAND = "lowland"
    MEDIUM = "medium"
    HIGHLAND = "highland"
    ALPINE = "alpine"


class LatitudeBand(str, Enum):
    SOUTHERN = "southern"
    MIDDLE = "middle"
    NORTHERN = "northern"


class LongitudeBand(str, Enum):
    WESTERN = "western"
    EASTERN = "eastern"


class TerrainGroup(str, Enum):
    FOREST = "forest"
    WATER = "water"
    OPEN = "open"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    UNKNOWN = "unknown"


@dataclass
class FeatureExtractionResult:
    track_id: str
    point_index: int
    features: Dict[str, Union[float, int, str, bool]]
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = {'track_id': self.track_id, 'point_index': self.point_index}
        result.update(self.features)
        return result



class TemporalFeatureExtractor:
    
    MONTH_TO_SEASON = {
        12: Season.WINTER, 1: Season.WINTER, 2: Season.WINTER,
        3: Season.SPRING, 4: Season.SPRING, 5: Season.SPRING,
        6: Season.SUMMER, 7: Season.SUMMER, 8: Season.SUMMER,
        9: Season.AUTUMN, 10: Season.AUTUMN, 11: Season.AUTUMN,
    }
    
    EXPEDITION_MONTHS = {6, 7, 8}
    
    @staticmethod
    def extract(date_str: str) -> Dict[str, Union[int, str, bool]]:
        try:
            date = pd.to_datetime(date_str)
            month = date.month
            year = date.year
            
            season = TemporalFeatureExtractor.MONTH_TO_SEASON.get(month, Season.WINTER)
            day_of_week = date.weekday()
            day_of_year = date.dayofyear
            quarter = (month - 1) // 3 + 1
            
            return {
                'T_month': month,
                'T_season': season.value,
                'T_day_of_week': day_of_week,
                'T_day_of_year': day_of_year,
                'T_year': year,
                'T_is_winter': 1 if season == Season.WINTER else 0,
                'T_is_expedition_month': 1 if month in TemporalFeatureExtractor.EXPEDITION_MONTHS else 0,
                'T_quarter': quarter,
            }
        except Exception as e:
            logger.warning(f"Error extracting temporal features from {date_str}: {e}")
            return {
                'T_month': None,
                'T_season': None,
                'T_day_of_week': None,
                'T_day_of_year': None,
                'T_year': None,
                'T_is_winter': None,
                'T_is_expedition_month': None,
                'T_quarter': None,
            }


class GeographicFeatureExtractor:
    
    SOUTHERN_LATITUDE_THRESHOLD = 55.0
    MIDDLE_LATITUDE_THRESHOLD = 60.0
    WESTERN_LONGITUDE_THRESHOLD = 70.0
    
    @staticmethod
    def extract(latitude: float, longitude: float, region: Optional[str] = None) -> Dict:
        try:
            features = {
                'G_latitude': latitude,
                'G_longitude': longitude,
                'G_region': region if region else 'Unknown',
            }
            
            if latitude < GeographicFeatureExtractor.SOUTHERN_LATITUDE_THRESHOLD:
                features['G_latitude_band'] = LatitudeBand.SOUTHERN.value
            elif latitude < GeographicFeatureExtractor.MIDDLE_LATITUDE_THRESHOLD:
                features['G_latitude_band'] = LatitudeBand.MIDDLE.value
            else:
                features['G_latitude_band'] = LatitudeBand.NORTHERN.value
            
            features['G_longitude_band'] = (
                LongitudeBand.WESTERN.value 
                if longitude < GeographicFeatureExtractor.WESTERN_LONGITUDE_THRESHOLD
                else LongitudeBand.EASTERN.value
            )
            
            utm_zone = int((longitude + 180) / 6) + 1
            features['G_utm_zone'] = utm_zone
            
            distance_to_pole_km = (90.0 - latitude) * 111.32
            features['G_distance_to_pole_km'] = round(distance_to_pole_km, 2)
            
            features['G_latitude_sin'] = round(math.sin(math.radians(latitude)), 4)
            features['G_longitude_sin'] = round(math.sin(math.radians(longitude)), 4)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting geographic features: {e}")
            return {
                'G_latitude': latitude,
                'G_longitude': longitude,
                'G_region': region,
                'G_latitude_band': None,
                'G_longitude_band': None,
                'G_utm_zone': None,
                'G_distance_to_pole_km': None,
                'G_latitude_sin': None,
                'G_longitude_sin': None,
            }


class MeteorologicalFeatureExtractor:
    
    TEMP_THRESHOLDS = {
        -15: TemperatureClass.VERY_COLD,
        -5: TemperatureClass.COLD,
        5: TemperatureClass.COOL,
        15: TemperatureClass.WARM,
        float('inf'): TemperatureClass.HOT,
    }
    
    @staticmethod
    def extract(temperature: Optional[float], 
                month: Optional[int] = None) -> Dict[str, Union[float, int, str, bool]]:
        try:
            features = {}
            
            if temperature is not None:
                features['M_temperature'] = round(temperature, 2)
                
                temp_class = TemperatureClass.HOT
                for threshold, tclass in MeteorologicalFeatureExtractor.TEMP_THRESHOLDS.items():
                    if temperature < threshold:
                        temp_class = tclass
                        break
                features['M_temperature_class'] = temp_class.value
                
                features['M_is_cold'] = 1 if temperature < -5 else 0
                features['M_is_freezing'] = 1 if temperature < 0 else 0
                features['M_is_hot'] = 1 if temperature > 20 else 0
                
            else:
                features['M_temperature'] = None
                features['M_temperature_class'] = None
                features['M_is_cold'] = None
                features['M_is_freezing'] = None
                features['M_is_hot'] = None
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting meteorological features: {e}")
            return {
                'M_temperature': None,
                'M_temperature_class': None,
                'M_is_cold': None,
                'M_is_freezing': None,
                'M_is_hot': None,
            }


class TerrainFeatureExtractor:
    
    FOREST_TYPES = {'forest', 'taiga', 'deciduous_forest', 'coniferous_forest'}
    WATER_TYPES = {'water', 'lake', 'river', 'stream', 'pond'}
    OPEN_TYPES = {'steppe', 'tundra', 'meadow', 'grassland', 'field', 'farmland'}
    MOUNTAIN_TYPES = {'rocks', 'cliff', 'stone', 'mountain', 'glacier', 'snowfield', 'ice'}
    URBAN_TYPES = {'settlement', 'village', 'town', 'building', 'urban', 'park', 'road_asphalt', 
                   'road_gravel', 'road_dirt', 'trail', 'path'}
    
    @staticmethod
    def extract(terrain_type: Optional[str], 
                altitude: Optional[float],
                terrain_confidence: Optional[float] = None) -> Dict:
        try:
            features = {
                'TR_terrain_type': terrain_type if terrain_type else 'unknown',
                'TR_terrain_confidence': round(terrain_confidence, 2) if terrain_confidence else None,
                'TR_altitude': altitude,
            }
            
            if altitude is not None:
                if altitude < 500:
                    features['TR_altitude_class'] = AltitudeClass.LOWLAND.value
                elif altitude < 2000:
                    features['TR_altitude_class'] = AltitudeClass.MEDIUM.value
                elif altitude < 5000:
                    features['TR_altitude_class'] = AltitudeClass.HIGHLAND.value
                else:
                    features['TR_altitude_class'] = AltitudeClass.ALPINE.value
            else:
                features['TR_altitude_class'] = None
            
            terrain_lower = (terrain_type or '').lower() if terrain_type else ''
            
            features['TR_is_forest'] = 1 if any(t in terrain_lower for t in TerrainFeatureExtractor.FOREST_TYPES) else 0
            features['TR_is_water'] = 1 if any(t in terrain_lower for t in TerrainFeatureExtractor.WATER_TYPES) else 0
            features['TR_is_open'] = 1 if any(t in terrain_lower for t in TerrainFeatureExtractor.OPEN_TYPES) else 0
            features['TR_is_mountain'] = 1 if any(t in terrain_lower for t in TerrainFeatureExtractor.MOUNTAIN_TYPES) else 0
            features['TR_is_urban'] = 1 if any(t in terrain_lower for t in TerrainFeatureExtractor.URBAN_TYPES) else 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting terrain features: {e}")
            return {
                'TR_terrain_type': terrain_type,
                'TR_terrain_confidence': terrain_confidence,
                'TR_altitude': altitude,
                'TR_altitude_class': None,
                'TR_is_forest': None,
                'TR_is_water': None,
                'TR_is_open': None,
                'TR_is_mountain': None,
                'TR_is_urban': None,
            }


class EnvironmentalFeatureExtractor:
    
    SHELTER_KEYWORDS = {'shelter', 'hut', 'cabin', 'refuge', 'hospice'}
    WATER_KEYWORDS = {'water_source', 'spring', 'well', 'river', 'stream', 'lake', 'pond'}
    LANDMARK_KEYWORDS = {'peak', 'summit', 'pass', 'col', 'saddle', 'ridge'}
    
    @staticmethod
    def extract(nearby_objects: Optional[List] = None) -> Dict[str, Union[int, float, bool]]:
        try:
            features = {
                'OBJ_count_nearby': 0,
                'OBJ_shelter_count': 0,
                'OBJ_water_count': 0,
                'OBJ_landmark_count': 0,
                'OBJ_nearest_distance_m': None,
                'OBJ_has_shelter': 0,
                'OBJ_has_water': 0,
                'OBJ_has_landmark': 0,
                'OBJ_poi_density_per_km2': 0.0,
            }
            
            if nearby_objects is None or len(nearby_objects) == 0:
                return features
            
            features['OBJ_count_nearby'] = len(nearby_objects)
            
            distances = []
            
            for obj in nearby_objects:
                try:
                    obj_type = None
                    obj_distance = None
                    
                    if hasattr(obj, 'poi_type'):
                        obj_type = (obj.poi_type.value if hasattr(obj.poi_type, 'value') 
                                   else str(obj.poi_type)).lower()
                        obj_distance = obj.distance_m
                    elif isinstance(obj, dict):
                        obj_type = obj.get('type', '').lower()
                        obj_distance = obj.get('distance_m')
                    
                    if obj_type and obj_distance:
                        distances.append(obj_distance)
                        
                        if any(k in obj_type for k in EnvironmentalFeatureExtractor.SHELTER_KEYWORDS):
                            features['OBJ_shelter_count'] += 1
                            features['OBJ_has_shelter'] = 1
                        
                        if any(k in obj_type for k in EnvironmentalFeatureExtractor.WATER_KEYWORDS):
                            features['OBJ_water_count'] += 1
                            features['OBJ_has_water'] = 1
                        
                        if any(k in obj_type for k in EnvironmentalFeatureExtractor.LANDMARK_KEYWORDS):
                            features['OBJ_landmark_count'] += 1
                            features['OBJ_has_landmark'] = 1
                
                except Exception as e:
                    logger.debug(f"Error processing object: {e}")
                    continue
            
            if distances:
                features['OBJ_nearest_distance_m'] = round(min(distances), 1)
            
            area_km2 = math.pi * (0.5 ** 2)
            features['OBJ_poi_density_per_km2'] = round(features['OBJ_count_nearby'] / area_km2, 2)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting environmental features: {e}")
            return {
                'OBJ_count_nearby': None,
                'OBJ_shelter_count': None,
                'OBJ_water_count': None,
                'OBJ_landmark_count': None,
                'OBJ_nearest_distance_m': None,
                'OBJ_has_shelter': None,
                'OBJ_has_water': None,
                'OBJ_has_landmark': None,
                'OBJ_poi_density_per_km2': None,
            }



class FeatureExtractor:
    
    def __init__(self):
        self.temporal_extractor = TemporalFeatureExtractor()
        self.geographic_extractor = GeographicFeatureExtractor()
        self.meteorological_extractor = MeteorologicalFeatureExtractor()
        self.terrain_extractor = TerrainFeatureExtractor()
        self.environmental_extractor = EnvironmentalFeatureExtractor()
        
        logger.info("[OK] FeatureExtractor Запуск")
    
    def extract_point_features(self, 
                              track_id: str,
                              point_index: int,
                              date: str,
                              latitude: float,
                              longitude: float,
                              altitude: Optional[float] = None,
                              temperature: Optional[float] = None,
                              region: Optional[str] = None,
                              terrain_type: Optional[str] = None,
                              terrain_confidence: Optional[float] = None,
                              step_frequency: Optional[float] = None,
                              nearby_objects: Optional[List] = None) -> FeatureExtractionResult:
        features = {}
        errors = []
        
        try:
            temp_date = pd.to_datetime(date)
            month = temp_date.month
            
            temporal_features = self.temporal_extractor.extract(date)
            features.update(temporal_features)
            
            geographic_features = self.geographic_extractor.extract(latitude, longitude, region)
            features.update(geographic_features)
            
            meteorological_features = self.meteorological_extractor.extract(temperature, month)
            features.update(meteorological_features)
            
            terrain_features = self.terrain_extractor.extract(terrain_type, altitude, terrain_confidence)
            features.update(terrain_features)
            
            environmental_features = self.environmental_extractor.extract(nearby_objects)
            features.update(environmental_features)
            
            if step_frequency is not None:
                features['TJ_step_frequency'] = round(step_frequency, 4)
            else:
                features['TJ_step_frequency'] = None
            
            non_null_fields = sum(1 for v in features.values() if v is not None)
            total_fields = len(features)
            features['Q_data_completeness'] = round(non_null_fields / total_fields, 3)
            
        except Exception as e:
            logger.error(f"Error extracting features for point {track_id}[{point_index}]: {e}")
            errors.append(str(e))
        
        return FeatureExtractionResult(
            track_id=track_id,
            point_index=point_index,
            features=features,
            errors=errors
        )
    
    def extract_batch_features(self, enriched_points_list: List[Dict]) -> pd.DataFrame:
        logger.info(f"Extracting features for {len(enriched_points_list)} points...")
        
        results = []
        errors_total = 0
        
        for i, point_dict in enumerate(enriched_points_list):
            try:
                result = self.extract_point_features(
                    track_id=point_dict.get('track_id', 'UNKNOWN'),
                    point_index=point_dict.get('point_index', i),
                    date=point_dict.get('date', '2004-01-01'),
                    latitude=point_dict.get('latitude', 55.7558),
                    longitude=point_dict.get('longitude', 37.6173),
                    altitude=point_dict.get('altitude'),
                    temperature=point_dict.get('temperature'),
                    region=point_dict.get('region'),
                    terrain_type=point_dict.get('terrain_type'),
                    terrain_confidence=point_dict.get('terrain_confidence'),
                    step_frequency=point_dict.get('step_frequency'),
                    nearby_objects=point_dict.get('nearby_objects'),
                )
                
                if result.errors:
                    errors_total += len(result.errors)
                
                results.append(result.to_dict())
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"  → Processed {i + 1}/{len(enriched_points_list)} points")
                
            except Exception as e:
                logger.error(f"Error processing point {i}: {e}")
                errors_total += 1
                continue
        
        if not results:
            logger.warning("No features extracted!")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        logger.info(f"[OK] Features extracted: {len(df)} points, {len(df.columns)} features")
        logger.info(f"   > Errors encountered: {errors_total}")
        logger.info(f"   > Features: {list(df.columns)[:10]}... (showing first 10)")
        
        return df
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Extracting features from DataFrame ({len(df)} rows)...")
        
        points_list = df.to_dict('records')
        return self.extract_batch_features(points_list)



def initialize_feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()

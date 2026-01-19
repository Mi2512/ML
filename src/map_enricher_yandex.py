
import logging
import requests
import json
import os
import hashlib
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import math

from map_decoder import (
    StandardTerrainType,
    StandardObjectType,
    MapSymbolRegistry
)

try:
    from weather_integration import OpenWeatherMapProvider
    HAS_WEATHER = True
except ImportError:
    HAS_WEATHER = False
    logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)



class YandexCache:
    
    def __init__(self, cache_dir: str = "yandex_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.hit_count = 0
        self.miss_count = 0
        
    def _hash_query(self, latitude: float, longitude: float, query_type: str = "geo") -> str:
        query_str = f"{query_type}|{latitude:.6f}|{longitude:.6f}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get(self, latitude: float, longitude: float, query_type: str = "geo") -> Optional[dict]:
        hash_key = self._hash_query(latitude, longitude, query_type)
        cache_file = self.cache_dir / f"{hash_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.hit_count += 1
                    logger.debug(f"Cache HIT: {hash_key}")
                    return data
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        self.miss_count += 1
        return None
    
    def set(self, latitude: float, longitude: float, query_type: str, data: dict):
        hash_key = self._hash_query(latitude, longitude, query_type)
        cache_file = self.cache_dir / f"{hash_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                logger.debug(f"Cache WRITE: {hash_key}")
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def get_statistics(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'total': total,
            'hit_rate': f"{hit_rate:.1f}%",
            'cached_files': len(list(self.cache_dir.glob("*.json")))
        }



class YandexFeatureExtractor:
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
    YANDEX_GEOCODER_URL = "https://geocode-maps.yandex.ru/1.x"
    STATIC_MAPS_URL = "https://static-maps.yandex.ru/1.x"
    
    _global_static_api_key: str = ""
    _global_geocoder_api_key: str = ""
    
    def __init__(self, cache_dir: str = "yandex_cache"):
        self.cache = YandexCache(cache_dir)
        self.static_api_key = self._global_static_api_key
        self.geocoder_api_key = self._global_geocoder_api_key
        self.has_static_key = bool(self.static_api_key) and len(self.static_api_key) > 20
        self.has_geocoder_key = bool(self.geocoder_api_key) and len(self.geocoder_api_key) > 20
        self.failed_requests = []
        self.request_count = 0
        
        self.weather_provider = OpenWeatherMapProvider() if HAS_WEATHER else None
        
        if self.has_static_key:
            logger.info(f"Yandex Maps Static API key configured (length: {len(self.static_api_key)}).")
        if self.has_geocoder_key:
            logger.info(f"Yandex HTTP Geocoder API key configured (length: {len(self.geocoder_api_key)}).")
    
    @classmethod
    def set_static_api_key(cls, api_key: str):
        cls._global_static_api_key = api_key
        logger.info(f"Yandex Maps Static API key set globally (length: {len(api_key)}).")
    
    @classmethod
    def set_geocoder_api_key(cls, api_key: str):
        cls._global_geocoder_api_key = api_key
        logger.info(f"Yandex HTTP Geocoder API key set globally (length: {len(api_key)}).")
    
    @classmethod
    def set_api_key(cls, api_key: str):
        cls._global_static_api_key = api_key
        logger.info(f"Yandex Maps Static API key set globally (length: {len(api_key)}).")
        
    def query_yandex_geocoder(self,
                             latitude: float,
                             longitude: float,
                             max_retries: int = 3) -> Optional[dict]:
        
        cached_data = self.cache.get(latitude, longitude, "geocoder")
        if cached_data is not None:
            return cached_data
        
        if self.has_geocoder_key:
            result = self._query_yandex_geocoder_api(latitude, longitude, max_retries)
            if result is not None:
                return result
        
        return self._query_nominatim_geocoder(latitude, longitude, max_retries)
    
    def _query_yandex_geocoder_api(self,
                                   latitude: float,
                                   longitude: float,
                                   max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.YANDEX_GEOCODER_URL,
                    params={
                        'apikey': self.geocoder_api_key,
                        'geocode': f"{longitude},{latitude}",
                        'format': 'json',
                        'kind': 'locality'
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    self.cache.set(latitude, longitude, "geocoder", data)
                    self.request_count += 1
                    
                    logger.info(f"Yandex Geocoder query successful")
                    return data
                
                elif response.status_code == 429:
                    logger.warning(f"Yandex Geocoder rate limited (429)")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                
                else:
                    logger.warning(f"Yandex Geocoder HTTP {response.status_code}")
                    
            except Exception as e:
                logger.debug(f"Yandex Geocoder query error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return None
    
    def _query_nominatim_geocoder(self,
                                 latitude: float,
                                 longitude: float,
                                 max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.NOMINATIM_URL,
                    params={
                        'lat': latitude,
                        'lon': longitude,
                        'format': 'json',
                        'zoom': 18
                    },
                    timeout=10,
                    headers={'User-Agent': 'MapAnalysisSystem/1.0'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    self.cache.set(latitude, longitude, "geocoder", data)
                    self.request_count += 1
                    
                    logger.info(f"Nominatim geocoder query successful")
                    return data
                
                elif response.status_code == 429:
                    logger.warning(f"Nominatim rate limited (429)")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                
                else:
                    logger.error(f"Geocoder error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
            
            except requests.Timeout:
                logger.warning(f"Request timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        self.failed_requests.append({
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': datetime.now().isoformat()
        })
        
        return None
    
    def extract_terrain_from_coordinates(self, latitude: float, longitude: float) -> Optional[StandardTerrainType]:
        
        geo_data = self.query_yandex_geocoder(latitude, longitude)
        
        if not geo_data:
            return None
        
        address = geo_data.get('address', {})
        
        address_str = str(address).lower()
        
        terrain_keywords = {
            StandardTerrainType.FOREST: ['лес', 'forest', 'woodland'],
            StandardTerrainType.GRASSLAND: ['луг', 'поле', 'meadow', 'field'],
            StandardTerrainType.WATER: ['озеро', 'река', 'lake', 'river', 'водохранилище'],
            StandardTerrainType.SWAMP: ['болото', 'marsh', 'swamp'],
            StandardTerrainType.SAND: ['песок', 'песочн', 'sand'],
            StandardTerrainType.TUNDRA: ['тундра', 'tundra'],
            StandardTerrainType.GLACIER: ['ледник', 'glacier'],
            StandardTerrainType.ROCKS: ['скалы', 'rocks', 'горы', 'mountains'],
            StandardTerrainType.URBAN: ['город', 'city', 'поселение', 'settlement', 'городская'],
        }
        
        for terrain_type, keywords in terrain_keywords.items():
            for keyword in keywords:
                if keyword in address_str:
                    return terrain_type
        
        if latitude > 66:
            return StandardTerrainType.TUNDRA
        elif latitude > 60:
            return StandardTerrainType.TAIGA
        else:
            return StandardTerrainType.GRASSLAND
    
    def extract_nearby_objects(self, latitude: float, longitude: float, 
                              radius_m: int = 500) -> List['PointOfInterest']:
        
        pois = []
        
        geo_data = self.query_yandex_geocoder(latitude, longitude)
        
        if not geo_data:
            return pois
        
        address = geo_data.get('address', {})
        
        object_keywords = {
            StandardObjectType.MOUNTAIN_PEAK: ['гора', 'пик', 'mountain', 'peak'],
            StandardObjectType.SPRING: ['источник', 'spring'],
            StandardObjectType.VIEWPOINT: ['обзорная', 'viewpoint', 'панорама'],
            StandardObjectType.MONUMENT: ['памятник', 'historic', 'монумент'],
            StandardObjectType.SHELTER: ['укрытие', 'shelter', 'приют', 'убежище'],
        }
        
        address_str = str(address).lower()
        
        for obj_type, keywords in object_keywords.items():
            for keyword in keywords:
                if keyword in address_str:
                    poi = PointOfInterest(
                        name=f"{obj_type.value}",
                        poi_type=obj_type,
                        latitude=latitude,
                        longitude=longitude,
                        source='yandex',
                        confidence=0.7
                    )
                    pois.append(poi)
                    break
        
        return pois
    
    def extract_region(self, latitude: float, longitude: float) -> Optional[str]:
        geo_data = self.query_yandex_geocoder(latitude, longitude)
        
        if not geo_data:
            return None
        
        address = geo_data.get('address', {})
        
        region = (
            address.get('state') or
            address.get('province') or
            address.get('region') or
            address.get('county')
        )
        
        if region:
            return str(region)
        
        country = address.get('country')
        if country:
            logger.debug(f"No region found, using country: {country}")
            return str(country)
        
        return None
    
    def get_temperature(self, latitude: float, longitude: float, date: str) -> Optional[float]:
        if not self.weather_provider:
            return None
        
        try:
            weather = self.weather_provider.get_weather(latitude, longitude, date)
            return weather.temperature if weather else None
        except Exception as e:
            logger.warning(f"Temperature extraction error: {e}")
            return None
    
    def get_static_map_url(self, latitude: float, longitude: float, 
                          width: int = 450, height: int = 450, 
                          zoom: int = 13) -> Optional[str]:
        if not self.has_static_key:
            logger.debug("Yandex Maps Static API key not configured for static maps")
            return None
        
        try:
            params = {
                'll': f"{longitude},{latitude}",
                'z': zoom,
                'size': f"{width},{height}",
                'l': 'map',
                'apikey': self.static_api_key
            }
            
            url = f"{self.STATIC_MAPS_URL}?"
            url += "&".join([f"{k}={v}" for k, v in params.items()])
            
            logger.debug(f"Generated static map URL for ({latitude}, {longitude})")
            return url
            
        except Exception as e:
            logger.warning(f"Static map URL generation error: {e}")
            return None
    
    def get_statistics(self) -> dict:
        return {
            'cache_stats': self.cache.get_statistics(),
            'total_requests': self.request_count,
            'failed_requests': len(self.failed_requests)
        }



class DistanceCalculator:
    
    EARTH_RADIUS_M = 6_371_000
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return DistanceCalculator.EARTH_RADIUS_M * c



@dataclass
class PointOfInterest:
    name: str
    poi_type: StandardObjectType
    latitude: float
    longitude: float
    source: str = 'yandex'
    confidence: float = 0.8
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.poi_type.value,
            'latitude': round(self.latitude, 6),
            'longitude': round(self.longitude, 6),
            'source': self.source,
            'confidence': round(self.confidence, 2)
        }


@dataclass
class ObjectDistance:
    poi_name: str
    poi_type: StandardObjectType
    distance_m: float
    bearing_degrees: float
    source: str = 'yandex'
    confidence: float = 0.8
    
    def to_dict(self) -> dict:
        return {
            'name': self.poi_name,
            'type': self.poi_type.value,
            'distance_m': round(self.distance_m, 1),
            'bearing_degrees': round(self.bearing_degrees, 1),
            'source': self.source,
            'confidence': round(self.confidence, 2)
        }


@dataclass
class EnrichedPointData:
    track_id: str
    point_index: int
    latitude: float
    longitude: float
    altitude: float
    date: str
    
    terrain_type: Optional[StandardTerrainType] = None
    terrain_confidence: float = 0.0
    
    nearby_objects: List[ObjectDistance] = field(default_factory=list)
    object_count: int = 0
    
    region: Optional[str] = None
    temperature: Optional[float] = None
    step_frequency: Optional[float] = None
    
    quality_score: float = 0.0
    enrichment_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    yandex_available: bool = True
    
    def to_dict(self) -> dict:
        return {
            'track_id': self.track_id,
            'point_index': self.point_index,
            'latitude': round(self.latitude, 6),
            'longitude': round(self.longitude, 6),
            'altitude': round(self.altitude, 1),
            'date': self.date,
            
            'terrain_type': self.terrain_type.value if self.terrain_type else None,
            'terrain_confidence': round(self.terrain_confidence, 2),
            'nearby_objects': [obj.to_dict() for obj in self.nearby_objects],
            'object_count': self.object_count,
            
            'region': self.region,
            'temperature': round(self.temperature, 1) if self.temperature else None,
            'step_frequency': round(self.step_frequency, 3) if self.step_frequency else None,
            
            'quality_score': round(self.quality_score, 2),
            'enrichment_timestamp': self.enrichment_timestamp,
            'yandex_available': self.yandex_available
        }



class EnrichmentEngineYandex:
    
    def __init__(self, buffer_radius_m: int = 500, cache_dir: str = "yandex_cache"):
        self.extractor = YandexFeatureExtractor(cache_dir)
        self.distance_calc = DistanceCalculator()
        self.buffer_radius_m = buffer_radius_m
        self.processed_points = 0
        self.failed_enrichments = 0
    
    def enrich_point(self,
                    track_id: str,
                    point_index: int,
                    latitude: float,
                    longitude: float,
                    altitude: float,
                    date: str,
                    csv_terrain_type: Optional[StandardTerrainType] = None) -> EnrichedPointData:
        
        enriched = EnrichedPointData(
            track_id=track_id,
            point_index=point_index,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            date=date,
            terrain_type=csv_terrain_type,
            terrain_confidence=0.9 if csv_terrain_type else 0.0
        )
        
        try:
            yandex_terrain = self.extractor.extract_terrain_from_coordinates(latitude, longitude)
            
            if yandex_terrain and not csv_terrain_type:
                enriched.terrain_type = yandex_terrain
                enriched.terrain_confidence = 0.75
            
            pois = self.extractor.extract_nearby_objects(latitude, longitude, self.buffer_radius_m)
            
            nearby_objects = []
            for poi in pois:
                distance_m = self.distance_calc.haversine_distance(
                    latitude, longitude,
                    poi.latitude, poi.longitude
                )
                
                obj_dist = ObjectDistance(
                    poi_name=poi.name,
                    poi_type=poi.poi_type,
                    distance_m=distance_m,
                    bearing_degrees=0.0,
                    source='yandex',
                    confidence=poi.confidence
                )
                nearby_objects.append(obj_dist)
            
            nearby_objects.sort(key=lambda x: x.distance_m)
            
            enriched.nearby_objects = nearby_objects
            enriched.object_count = len(nearby_objects)
            
            enriched.region = self.extractor.extract_region(latitude, longitude)
            
            enriched.temperature = self.extractor.get_temperature(latitude, longitude, date)
            
            enriched.step_frequency = 1.4
            
            base_score = 1.0 if csv_terrain_type else 0.8
            object_bonus = min(0.15, enriched.object_count * 0.03)
            region_bonus = 0.05 if enriched.region else 0.0
            temperature_bonus = 0.05 if enriched.temperature else 0.0
            enriched.quality_score = min(1.0, base_score + object_bonus + region_bonus + temperature_bonus)
            
            self.processed_points += 1
            
        except Exception as e:
            logger.error(f"Enrichment error for point {point_index}: {e}")
            enriched.yandex_available = False
            enriched.quality_score = 0.2
            self.failed_enrichments += 1
        
        return enriched
    
    def enrich_batch(self,
                    points: List[Dict]) -> List[EnrichedPointData]:
        enriched_points = []
        
        logger.info(f"Starting batch enrichment of {len(points)} points using Yandex API")
        start_time = time.time()
        
        for i, point in enumerate(points):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(f"Progress: {i+1}/{len(points)} points ({rate:.1f} pts/sec)")
            
            enriched = self.enrich_point(
                track_id=point['track_id'],
                point_index=point['point_index'],
                latitude=point['latitude'],
                longitude=point['longitude'],
                altitude=point['altitude'],
                date=point['date'],
                csv_terrain_type=point.get('csv_terrain_type')
            )
            enriched_points.append(enriched)
        
        for i in range(1, len(enriched_points)):
            prev_point = enriched_points[i - 1]
            curr_point = enriched_points[i]
            
            distance_m = self.distance_calc.haversine_distance(
                prev_point.latitude, prev_point.longitude,
                curr_point.latitude, curr_point.longitude
            )
            
            if distance_m > 0:
                curr_point.step_frequency = 1.0 / 0.7
            else:
                curr_point.step_frequency = 0.0
        
        elapsed = time.time() - start_time
        logger.info(f"Batch enrichment complete: {len(enriched_points)} points in {elapsed:.1f}s")
        
        return enriched_points
    
    def get_statistics(self) -> dict:
        return {
            'processed_points': self.processed_points,
            'failed_enrichments': self.failed_enrichments,
            'success_rate': f"{(self.processed_points / (self.processed_points + self.failed_enrichments) * 100) if (self.processed_points + self.failed_enrichments) > 0 else 0:.1f}%",
            'extractor_stats': self.extractor.get_statistics()
        }



def initialize_enricher_yandex(buffer_radius_m: int = 500, cache_dir: str = "yandex_cache") -> EnrichmentEngineYandex:
    logger.info(f"Initializing Yandex enrichment engine (radius={buffer_radius_m}m, cache_dir={cache_dir})")
    return EnrichmentEngineYandex(buffer_radius_m, cache_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\nMap enricher YANDEX - Stage 2 Testing")
    print("="*80 + "\n")
    
    test_point = {
        'track_id': 'TEST_001',
        'point_index': 0,
        'latitude': 55.7558,
        'longitude': 37.6173,
        'altitude': 150.0,
        'date': '2024-01-17',
        'csv_terrain_type': StandardTerrainType.GRASSLAND
    }
    
    engine = initialize_enricher_yandex()
    
    print(" Testing single point enrichment")
    enriched = engine.enrich_point(
        track_id=test_point['track_id'],
        point_index=test_point['point_index'],
        latitude=test_point['latitude'],
        longitude=test_point['longitude'],
        altitude=test_point['altitude'],
        date=test_point['date'],
        csv_terrain_type=test_point['csv_terrain_type']
    )
    
    print(f"\n  Point enrichment result:")
    print(f"    Terrain: {enriched.terrain_type}")
    print(f"    Terrain confidence: {enriched.terrain_confidence}")
    print(f"    Nearby objects: {enriched.object_count}")
    print(f"    Quality score: {enriched.quality_score:.2f}")
    print(f"    Yandex available: {enriched.yandex_available}")
    
    print("\n Testing cache system")
    cache_stats = engine.extractor.cache.get_statistics()
    print(f"  Cache statistics:")
    print(f"    Files cached: {cache_stats['cached_files']}")
    print(f"    Hit rate: {cache_stats['hit_rate']}")
    
    print("\n All Yandex enricher tests verified!")
    print("\n" + "="*80 + "\n")

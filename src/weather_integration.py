
import logging
import requests
import json
import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    weather_code: str
    weather_description: str
    source: str
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            'temperature': round(self.temperature, 1),
            'humidity': round(self.humidity, 1),
            'pressure': round(self.pressure, 1),
            'wind_speed': round(self.wind_speed, 2),
            'weather_code': self.weather_code,
            'weather_description': self.weather_description,
            'source': self.source,
            'confidence': round(self.confidence, 2)
        }


class WeatherCache:
    
    def __init__(self, cache_dir: str = "weather_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _hash_query(self, latitude: float, longitude: float, date: str) -> str:
        query_str = f"weather|{latitude:.6f}|{longitude:.6f}|{date}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def get(self, latitude: float, longitude: float, date: str) -> Optional[WeatherData]:
        hash_key = self._hash_query(latitude, longitude, date)
        cache_file = self.cache_dir / f"{hash_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Weather cache HIT: {hash_key}")
                    return WeatherData(**data)
            except Exception as e:
                logger.warning(f"Weather cache read error: {e}")
        
        return None
    
    def set(self, latitude: float, longitude: float, date: str, weather: WeatherData):
        hash_key = self._hash_query(latitude, longitude, date)
        cache_file = self.cache_dir / f"{hash_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(weather.to_dict(), f, indent=2)
                logger.debug(f"Weather cache WRITE: {hash_key}")
        except Exception as e:
            logger.error(f"Weather cache write error: {e}")


class OpenWeatherMapProvider:
    
    _global_api_key: str = ""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "weather_cache"):
        self.api_key = api_key or self._global_api_key or os.environ.get('OPENWEATHERMAP_API_KEY', '')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.cache = WeatherCache(cache_dir)
        self.has_valid_key = bool(self.api_key) and len(self.api_key) > 10
        
        if not self.has_valid_key:
            logger.warning("OpenWeatherMap Нет API ключа. Using estimation fallback")
        else:
            logger.info(f"OpenWeatherMap API key configured (length: {len(self.api_key)}).")
    
    @classmethod
    def set_api_key(cls, api_key: str):
        cls._global_api_key = api_key
        logger.info(f"OpenWeatherMap API key set globally (length: {len(api_key)}).")
    
    def get_weather(self, latitude: float, longitude: float, date: str) -> WeatherData:
        
        cached = self.cache.get(latitude, longitude, date)
        if cached:
            return cached
        
        if self.has_valid_key:
            weather = self._query_openweathermap(latitude, longitude)
            if weather:
                self.cache.set(latitude, longitude, date, weather)
                return weather
        
        weather = self._estimate_weather(latitude, longitude, date)
        self.cache.set(latitude, longitude, date, weather)
        return weather
    
    def _query_openweathermap(self, latitude: float, longitude: float) -> Optional[WeatherData]:
        try:
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            logger.debug(f"Querying OpenWeatherMap API for ({latitude}, {longitude})")
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            main = data.get('main', {})
            weather = data.get('weather', [{}])[0]
            wind = data.get('wind', {})
            
            logger.info(f"OpenWeatherMap API returned data for ({latitude}, {longitude})")
            return WeatherData(
                temperature=float(main.get('temp', 15.0)),
                humidity=float(main.get('humidity', 50.0)),
                pressure=float(main.get('pressure', 1013.0)),
                wind_speed=float(wind.get('speed', 0.0)),
                weather_code=weather.get('main', 'Unknown'),
                weather_description=weather.get('description', 'No description'),
                source='openweathermap',
                confidence=0.85
            )
            
        except requests.RequestException as e:
            logger.warning(f"OpenWeatherMap API HTTP error ({latitude}, {longitude}): {e}")
            return None
        except Exception as e:
            logger.warning(f"OpenWeatherMap API error ({latitude}, {longitude}): {e}")
            return None
    
    def _estimate_weather(self, latitude: float, longitude: float, date: str) -> WeatherData:
        try:
            date_obj = datetime.fromisoformat(date)
            month = date_obj.month
        except:
            month = 6
        
        if latitude > 66:
            base_temp = -15.0 if month in [12, 1, 2] else -2.0
            humidity = 75
            wind_speed = 6.0
            weather_code = "13d" if month in [12, 1, 2] else "01d"
        elif latitude > 60:
            base_temp = -8.0 if month in [12, 1, 2] else 12.0
            humidity = 70
            wind_speed = 4.5
            weather_code = "13d" if month in [12, 1, 2] else "04d"
        elif latitude > 45:
            base_temp = -2.0 if month in [12, 1, 2] else 18.0
            humidity = 65
            wind_speed = 3.5
            weather_code = "10d" if month in [11, 12, 1, 2] else "04d"
        else:
            base_temp = 20.0 if month in [12, 1, 2] else 28.0
            humidity = 70 if month in [6, 7, 8] else 55
            wind_speed = 2.5
            weather_code = "01d"
        
        longitude_factor = (longitude % 30) / 30 * 4.0 - 2.0
        base_temp += longitude_factor
        
        return WeatherData(
            temperature=base_temp,
            humidity=humidity,
            pressure=1013.0,
            wind_speed=wind_speed,
            weather_code=weather_code,
            weather_description=self._code_to_description(weather_code),
            source='estimation',
            confidence=0.45
        )
    
    @staticmethod
    def _code_to_description(code: str) -> str:
        code_map = {
            '01d': 'Sunny day',
            '01n': 'Clear night',
            '02d': 'Partly cloudy day',
            '02n': 'Partly cloudy night',
            '03d': 'Cloudy day',
            '03n': 'Cloudy night',
            '04d': 'Overcast day',
            '04n': 'Overcast night',
            '09d': 'Drizzle day',
            '09n': 'Drizzle night',
            '10d': 'Rainy day',
            '10n': 'Rainy night',
            '11d': 'Thunderstorm day',
            '11n': 'Thunderstorm night',
            '13d': 'Snowy day',
            '13n': 'Snowy night',
            '50d': 'Fog day',
            '50n': 'Fog night',
        }
        return code_map.get(code, 'Unknown')
    
    def get_statistics(self) -> dict:
        cache_files = len(list(self.cache.cache_dir.glob("*.json")))
        return {
            'has_api_key': self.has_valid_key,
            'cached_records': cache_files,
            'api_url': self.base_url
        }


def get_batch_weather(points: list, api_key: Optional[str] = None) -> dict:
    provider = OpenWeatherMapProvider(api_key)
    results = {}
    
    for i, point in enumerate(points):
        weather = provider.get_weather(
            point.get('latitude', 0),
            point.get('longitude', 0),
            point.get('date', datetime.now().isoformat())
        )
        results[i] = weather
    
    return results

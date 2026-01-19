
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class WeatherLoader:
    
    def __init__(self):
        logger.info("WeatherLoader Запуск")
        try:
            from weather_integration import WeatherIntegration
            self.weather = WeatherIntegration()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not import weather integration: {e}")
            self.weather = None
    
    def get_weather(self, lat: float, lon: float, 
                   timestamp: Optional[datetime] = None) -> Dict:
        if self.weather:
            return self.weather.get_weather(lat, lon, timestamp)
        
        return {
            "status": "error",
            "message": "Weather service unavailable"
        }
    
    def get_historical_weather(self, lat: float, lon: float, 
                              date: datetime) -> Dict:
        if self.weather and hasattr(self.weather, 'get_historical_weather'):
            return self.weather.get_historical_weather(lat, lon, date)
        
        return {
            "status": "error",
            "message": "Historical weather not available"
        }
    
    def get_weather_for_track(self, track_points: List[Dict]) -> List[Dict]:
        enriched_points = []
        
        for point in track_points:
            lat = point.get("latitude", 0)
            lon = point.get("longitude", 0)
            timestamp = point.get("timestamp")
            
            weather = self.get_weather(lat, lon, timestamp)
            
            enriched_point = {**point, "weather": weather}
            enriched_points.append(enriched_point)
        
        return enriched_points
    
    def validate_weather_data(self, weather_dict: Dict) -> bool:
        required_fields = ["temperature", "humidity", "pressure"]
        return all(field in weather_dict for field in required_fields)


WeatherIntegrationLoader = WeatherLoader

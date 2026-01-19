
import logging
import re
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

from map_decoder import MapSourceType

logger = logging.getLogger(__name__)


@dataclass
class MapSourceDetectionResult:
    source_type: Optional[MapSourceType]
    detection_method: str
    confidence: float
    indicators: List[str]
    alternative_sources: Dict[MapSourceType, float]
    
    def to_dict(self) -> dict:
        return {
            'source_type': self.source_type.value if self.source_type else None,
            'detection_method': self.detection_method,
            'confidence': round(self.confidence, 2),
            'indicators': self.indicators,
            'alternative_sources': {
                k.value: round(v, 2) for k, v in self.alternative_sources.items()
            }
        }


class MapSourceDetector:
    
    URL_PATTERNS = {
        MapSourceType.GOOGLE_MAPS: [
            r'maps\.google\.',
            r'google\.com/maps',
            r'googlemaps',
            r'maps\.gstatic\.com',
            r'googleapis\.com.*maps',
        ],
        MapSourceType.YANDEX_MAPS: [
            r'yandex\.',
            r'yandex\.ru/maps',
            r'yandex\.com/maps',
            r'api\.yandex\.com.*maps',
            r'maps\.yandex\.ru',
        ],
        MapSourceType.OPENSTREETMAP: [
            r'openstreetmap',
            r'osm\.org',
            r'overpass-api',
            r'nominatim',
            r'mapbox\.com',
        ],
        MapSourceType.STAMEN_TERRAIN: [
            r'stamen',
            r'stamen\.com',
            r'tile\.stamen\.com',
            r'a\.tile\.openstreetmap',
        ],
    }
    
    STYLE_INDICATORS = {
        MapSourceType.GOOGLE_MAPS: {
            'keywords': [
                'google.maps',
                'google-map',
                'Maps JavaScript API',
                '#4285F4',
                'google.maps.Map',
            ],
            'priority': 10
        },
        MapSourceType.YANDEX_MAPS: {
            'keywords': [
                'ymaps',
                'yandex.map',
                'Yandex Maps API',
                '#FF9900',
                'ymaps.Map',
            ],
            'priority': 10
        },
        MapSourceType.OPENSTREETMAP: {
            'keywords': [
                'leaflet',
                'openstreetmap',
                'osm',
                'overpass',
                'nominatim',
                'L.map',
            ],
            'priority': 8
        },
        MapSourceType.STAMEN_TERRAIN: {
            'keywords': [
                'stamen',
                'terrain',
                'toner',
                'stamen.com',
                'tile.stamen.com',
            ],
            'priority': 9
        },
    }
    
    METADATA_PATTERNS = {
        MapSourceType.GOOGLE_MAPS: {
            'server_patterns': [r'GSE', r'gws'],
            'js_patterns': [r'maps\.google\.', r'googleapis\.com'],
            'priority': 9
        },
        MapSourceType.YANDEX_MAPS: {
            'server_patterns': [r'Yandex', r'YandexBot'],
            'js_patterns': [r'yandex\.', r'api\.yandex\.'],
            'priority': 9
        },
        MapSourceType.OPENSTREETMAP: {
            'server_patterns': [r'Apache', r'nginx'],
            'js_patterns': [r'leaflet', r'openstreetmap'],
            'priority': 7
        },
    }
    
    def __init__(self):
        pass
    
    def detect_from_url(self, url: str) -> MapSourceDetectionResult:
        indicators = []
        scores = {}
        
        url_lower = url.lower()
        
        for source_type, patterns in self.URL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    indicators.append(f"URL pattern: {pattern}")
                    scores[source_type] = scores.get(source_type, 0) + 1
        
        if scores:
            best_source = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_source] / 3)
            
            alternatives = {
                src: min(1.0, score / 3) for src, score in scores.items() 
                if src != best_source
            }
            
            return MapSourceDetectionResult(
                source_type=best_source,
                detection_method='url',
                confidence=confidence,
                indicators=indicators,
                alternative_sources=alternatives
            )
        
        return MapSourceDetectionResult(
            source_type=None,
            detection_method='url',
            confidence=0.0,
            indicators=indicators,
            alternative_sources={}
        )
    
    def detect_from_html(self, html_content: str) -> MapSourceDetectionResult:
        indicators = []
        scores = {}
        
        html_lower = html_content.lower()
        
        for source_type, style_info in self.STYLE_INDICATORS.items():
            for keyword in style_info['keywords']:
                keyword_lower = keyword.lower()
                count = html_lower.count(keyword_lower)
                if count > 0:
                    indicators.append(f"Found keyword: {keyword} ({count} times)")
                    scores[source_type] = scores.get(source_type, 0) + count
        
        if scores:
            best_source = max(scores, key=scores.get)
            max_score = max(scores.values())
            confidence = min(1.0, max_score / 5)
            
            alternatives = {
                src: min(1.0, score / 5) for src, score in scores.items() 
                if src != best_source
            }
            
            return MapSourceDetectionResult(
                source_type=best_source,
                detection_method='visual',
                confidence=confidence,
                indicators=indicators,
                alternative_sources=alternatives
            )
        
        return MapSourceDetectionResult(
            source_type=None,
            detection_method='visual',
            confidence=0.0,
            indicators=indicators,
            alternative_sources={}
        )
    
    def detect_from_image_style(self, image_characteristics: Dict) -> MapSourceDetectionResult:
        indicators = []
        scores = {}
        
        dominant_color = image_characteristics.get('dominant_color_rgb', None)
        if dominant_color:
            r, g, b = dominant_color
            
            if 50 < r < 100 and 120 < g < 150 and 200 < b < 255:
                indicators.append("Detected Google blue (#4285F4)")
                scores[MapSourceType.GOOGLE_MAPS] = scores.get(MapSourceType.GOOGLE_MAPS, 0) + 2
            
            if 200 < r < 255 and 120 < g < 170 and 0 < b < 50:
                indicators.append("Detected Yandex orange (#FF9900)")
                scores[MapSourceType.YANDEX_MAPS] = scores.get(MapSourceType.YANDEX_MAPS, 0) + 2
            
            if 100 < r < 200 and 100 < g < 200 and 100 < b < 200:
                indicators.append("Detected neutral gray (typical of OSM)")
                scores[MapSourceType.OPENSTREETMAP] = scores.get(MapSourceType.OPENSTREETMAP, 0) + 1
        
        if image_characteristics.get('style_indicators'):
            for indicator in image_characteristics['style_indicators']:
                for source_type, style_info in self.STYLE_INDICATORS.items():
                    if indicator.lower() in ' '.join(style_info['keywords']).lower():
                        indicators.append(f"Style match: {indicator}")
                        scores[source_type] = scores.get(source_type, 0) + 1
        
        if scores:
            best_source = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_source] / 3)
            
            alternatives = {
                src: min(1.0, score / 3) for src, score in scores.items() 
                if src != best_source
            }
            
            return MapSourceDetectionResult(
                source_type=best_source,
                detection_method='visual',
                confidence=confidence,
                indicators=indicators,
                alternative_sources=alternatives
            )
        
        return MapSourceDetectionResult(
            source_type=None,
            detection_method='visual',
            confidence=0.0,
            indicators=indicators,
            alternative_sources={}
        )
    
    def detect_from_headers(self, headers: Dict[str, str]) -> MapSourceDetectionResult:
        indicators = []
        scores = {}
        
        for source_type, metadata_info in self.METADATA_PATTERNS.items():
            server = headers.get('Server', '').lower()
            for pattern in metadata_info['server_patterns']:
                if re.search(pattern.lower(), server):
                    indicators.append(f"Server pattern: {pattern}")
                    scores[source_type] = scores.get(source_type, 0) + 1
            
            content_type = headers.get('Content-Type', '').lower()
            for pattern in metadata_info['js_patterns']:
                if pattern.lower() in content_type:
                    indicators.append(f"Content pattern: {pattern}")
                    scores[source_type] = scores.get(source_type, 0) + 1
        
        if scores:
            best_source = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_source] / 2)
            
            alternatives = {
                src: min(1.0, score / 2) for src, score in scores.items() 
                if src != best_source
            }
            
            return MapSourceDetectionResult(
                source_type=best_source,
                detection_method='metadata',
                confidence=confidence,
                indicators=indicators,
                alternative_sources=alternatives
            )
        
        return MapSourceDetectionResult(
            source_type=None,
            detection_method='metadata',
            confidence=0.0,
            indicators=indicators,
            alternative_sources={}
        )
    
    def detect(self, 
               url: Optional[str] = None,
               html_content: Optional[str] = None,
               headers: Optional[Dict[str, str]] = None,
               image_characteristics: Optional[Dict] = None) -> MapSourceDetectionResult:
        results = []
        
        if url:
            results.append(self.detect_from_url(url))
        
        if html_content:
            results.append(self.detect_from_html(html_content))
        
        if headers:
            results.append(self.detect_from_headers(headers))
        
        if image_characteristics:
            results.append(self.detect_from_image_style(image_characteristics))
        
        if not results:
            return MapSourceDetectionResult(
                source_type=None,
                detection_method='unknown',
                confidence=0.0,
                indicators=[],
                alternative_sources={}
            )
        
        combined_scores = {}
        all_indicators = []
        detection_methods = set()
        
        for result in results:
            if result.source_type:
                combined_scores[result.source_type] = combined_scores.get(result.source_type, 0) + result.confidence
            all_indicators.extend(result.indicators)
            detection_methods.add(result.detection_method)
        
        if combined_scores:
            best_source = max(combined_scores, key=combined_scores.get)
            avg_confidence = combined_scores[best_source] / len(results)
            
            alternatives = {
                src: score / len(results) for src, score in combined_scores.items() 
                if src != best_source
            }
            
            return MapSourceDetectionResult(
                source_type=best_source,
                detection_method='+'.join(sorted(detection_methods)),
                confidence=min(1.0, avg_confidence),
                indicators=all_indicators,
                alternative_sources=alternatives
            )
        
        return MapSourceDetectionResult(
            source_type=None,
            detection_method='+'.join(sorted(detection_methods)),
            confidence=0.0,
            indicators=all_indicators,
            alternative_sources={}
        )

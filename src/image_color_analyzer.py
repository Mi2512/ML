
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logging.warning("OpenCV not available. Image analysis will be disabled.")

from map_decoder import StandardTerrainType, MapSourceType

logger = logging.getLogger(__name__)


@dataclass
class ColorAnalysisResult:
    dominant_color_rgb: Tuple[int, int, int]
    dominant_terrain: Optional[StandardTerrainType]
    color_histogram: Dict[str, float]
    map_source: Optional[MapSourceType]
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            'dominant_color_rgb': self.dominant_color_rgb,
            'dominant_terrain': self.dominant_terrain.value if self.dominant_terrain else None,
            'color_histogram': {k: round(v, 2) for k, v in self.color_histogram.items()},
            'map_source': self.map_source.value if self.map_source else None,
            'confidence': round(self.confidence, 2)
        }


class ImageColorAnalyzer:
    
    TERRAIN_COLOR_RANGES = {
        StandardTerrainType.FOREST: {
            'lower': np.array([20, 100, 20]),
            'upper': np.array([100, 200, 100]),
            'priority': 10
        },
        StandardTerrainType.GRASSLAND: {
            'lower': np.array([150, 180, 80]),
            'upper': np.array([220, 255, 150]),
            'priority': 9
        },
        StandardTerrainType.WATER: {
            'lower': np.array([50, 120, 180]),
            'upper': np.array([150, 200, 255]),
            'priority': 10
        },
        StandardTerrainType.URBAN: {
            'lower': np.array([180, 180, 180]),
            'upper': np.array([240, 220, 200]),
            'priority': 8
        },
        StandardTerrainType.ROCKS: {
            'lower': np.array([100, 100, 100]),
            'upper': np.array([180, 160, 140]),
            'priority': 7
        },
        StandardTerrainType.SAND: {
            'lower': np.array([200, 190, 140]),
            'upper': np.array([255, 240, 180]),
            'priority': 8
        },
        StandardTerrainType.SWAMP: {
            'lower': np.array([100, 140, 100]),
            'upper': np.array([160, 180, 140]),
            'priority': 6
        },
        StandardTerrainType.GLACIER: {
            'lower': np.array([200, 220, 255]),
            'upper': np.array([255, 255, 255]),
            'priority': 9
        },
    }
    
    MAP_SOURCE_SIGNATURES = {
        MapSourceType.YANDEX_MAPS: {
            'primary_color': (255, 153, 0),
            'tolerance': 35,
            'min_percentage': 0.08,
            'style_indicators': ['orange_roads', 'cyrillic_labels']
        },
    }
    
    def __init__(self):
        if not HAS_OPENCV:
            logger.warning("OpenCV not available. Image analysis features disabled")
        self.has_opencv = HAS_OPENCV
    
    def analyze_image(self, image_path: str) -> ColorAnalysisResult:
        
        if not self.has_opencv:
            logger.warning("OpenCV not available. Returning empty analysis")
            return ColorAnalysisResult(
                dominant_color_rgb=(128, 128, 128),
                dominant_terrain=None,
                color_histogram={},
                map_source=None,
                confidence=0.0
            )
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return self._empty_result()
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            dominant_color = self._get_dominant_color(image_rgb)
            
            terrain, terrain_confidence = self._analyze_terrain(image_rgb)
            
            map_source, source_confidence = self._detect_map_source(image_rgb)
            
            histogram = self._build_color_histogram(image_rgb)
            
            overall_confidence = (terrain_confidence + source_confidence) / 2 if terrain else source_confidence
            
            return ColorAnalysisResult(
                dominant_color_rgb=dominant_color,
                dominant_terrain=terrain,
                color_histogram=histogram,
                map_source=map_source,
                confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return self._empty_result()
    
    def _get_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        
        pixels = image.reshape(-1, 3)
        
        if HAS_OPENCV:
            pixels_float = np.float32(pixels)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, _, centers = cv2.kmeans(pixels_float, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            dominant = tuple(map(int, centers[0]))
            return dominant
        else:
            return tuple(map(int, np.mean(pixels, axis=0)))
    
    def _analyze_terrain(self, image: np.ndarray) -> Tuple[Optional[StandardTerrainType], float]:
        
        if not HAS_OPENCV:
            return None, 0.0
        
        best_terrain = None
        best_score = 0.0
        
        for terrain_type, color_range in self.TERRAIN_COLOR_RANGES.items():
            mask = cv2.inRange(
                image,
                color_range['lower'],
                color_range['upper']
            )
            
            percentage = np.count_nonzero(mask) / mask.size
            
            score = percentage * (color_range['priority'] / 10.0)
            
            if score > best_score:
                best_score = score
                best_terrain = terrain_type
        
        confidence = min(1.0, best_score * 2)
        
        return best_terrain, confidence
    
    def _detect_map_source(self, image: np.ndarray) -> Tuple[Optional[MapSourceType], float]:
        
        if not HAS_OPENCV:
            return None, 0.0
        
        best_source = None
        best_score = 0.0
        
        for source_type, signature in self.MAP_SOURCE_SIGNATURES.items():
            primary_rgb = signature['primary_color']
            tolerance = signature['tolerance']
            
            lower = np.array([max(0, c - tolerance) for c in primary_rgb])
            upper = np.array([min(255, c + tolerance) for c in primary_rgb])
            
            mask = cv2.inRange(image, lower, upper)
            
            percentage = np.count_nonzero(mask) / mask.size
            
            if percentage >= signature['min_percentage']:
                score = percentage / signature['min_percentage']
                
                if score > best_score:
                    best_score = score
                    best_source = source_type
        
        confidence = min(1.0, best_score * 0.8)
        
        return best_source, confidence
    
    def _build_color_histogram(self, image: np.ndarray) -> Dict[str, float]:
        
        histogram = {
            'red': 0.0,
            'green': 0.0,
            'blue': 0.0,
            'yellow': 0.0,
            'gray': 0.0,
            'brown': 0.0,
            'other': 0.0,
        }
        
        for y in range(0, image.shape[0], max(1, image.shape[0] // 100)):
            for x in range(0, image.shape[1], max(1, image.shape[1] // 100)):
                r, g, b = image[y, x]
                
                if r > 150 and g < 100 and b < 100:
                    histogram['red'] += 1
                elif g > 150 and r < 100 and b < 100:
                    histogram['green'] += 1
                elif b > 150 and r < 100 and g < 100:
                    histogram['blue'] += 1
                elif r > 150 and g > 150 and b < 100:
                    histogram['yellow'] += 1
                elif r > 100 and g > 50 and b < 100:
                    histogram['brown'] += 1
                elif abs(int(r) - int(g)) < 30 and abs(int(g) - int(b)) < 30:
                    histogram['gray'] += 1
                else:
                    histogram['other'] += 1
        
        total = sum(histogram.values())
        if total > 0:
            for key in histogram:
                histogram[key] = histogram[key] / total * 100
        
        return histogram
    
    def _empty_result(self) -> ColorAnalysisResult:
        return ColorAnalysisResult(
            dominant_color_rgb=(128, 128, 128),
            dominant_terrain=None,
            color_histogram={},
            map_source=None,
            confidence=0.0
        )


def analyze_map_images(image_paths: List[str]) -> Dict[str, ColorAnalysisResult]:
    analyzer = ImageColorAnalyzer()
    results = {}
    
    for path in image_paths:
        results[path] = analyzer.analyze_image(path)
    
    return results


import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import json

logger = logging.getLogger(__name__)



class StandardTerrainType(Enum):
    
    FOREST = "лес"
    TAIGA = "тайга"
    DECIDUOUS_FOREST = "лиственный_лес"
    CONIFEROUS_FOREST = "хвойный_лес"
    
    SWAMP = "болото"
    BOG = "болото_верховое"
    FEN = "болото_низинное"
    MARSH = "болото_травяное"
    
    STEPPE = "степь"
    TUNDRA = "тундра"
    MEADOW = "луг"
    GRASSLAND = "травянистая_местность"
    
    RIVER = "река"
    LAKE = "озеро"
    STREAM = "ручей"
    WATER = "водоём"
    
    ROCKS = "скалы"
    CLIFF = "утёс"
    STONE = "каменистая_местность"
    
    GLACIER = "ледник"
    SNOWFIELD = "снежное_поле"
    ICE = "ледниковое_поле"
    
    ROAD_ASPHALT = "дорога_асфальт"
    ROAD_GRAVEL = "дорога_грунтовая"
    ROAD_DIRT = "дорога_грунт"
    TRAIL = "тропа"
    PATH = "дорога_пешеходная"
    
    SETTLEMENT = "посёлок"
    VILLAGE = "деревня"
    TOWN = "город"
    BUILDING = "здание"
    
    URBAN = "городская_территория"
    PARK = "парк"
    
    FIELD = "поле"
    FARMLAND = "сельскохозяйственная_земля"
    
    SAND = "песок"
    MOUNTAIN = "гора"
    UNKNOWN = "неизвестно"


class StandardObjectType(Enum):
    
    MOUNTAIN_PEAK = "горная_вершина"
    MOUNTAIN_PASS = "перевал"
    RIDGE = "хребет"
    VALLEY = "долина"
    GORGE = "ущелье"
    WATERFALL = "водопад"
    SPRING = "источник_воды"
    LAKE = "озеро"
    RIVER = "река"
    GLACIER = "ледник"
    CAVE = "пещера"
    CLIFF = "утёс"
    ISLAND = "остров"
    PENINSULA = "полуостров"
    BAY = "залив"
    
    SETTLEMENT = "посёлок"
    VILLAGE = "деревня"
    TOWN = "город"
    TRAIN_STATION = "железнодорожная_станция"
    BUS_STOP = "автобусная_остановка"
    SHELTER = "приют"
    HUT = "избушка"
    CAMP = "лагерь"
    BRIDGE = "мост"
    TUNNEL = "туннель"
    
    CAFE = "кафе"
    RESTAURANT = "ресторан"
    SHOP = "магазин"
    HOSPITAL = "больница"
    CLINIC = "поликлиника"
    PHARMACY = "аптека"
    HOTEL = "гостиница"
    HOSTEL = "хостел"
    PARKING = "парковка"
    TOILET = "туалет"
    
    VIEWPOINT = "обзорная_площадка"
    MONUMENT = "памятник"
    HISTORICAL_SITE = "исторический_объект"
    LIGHTHOUSE = "маяк"
    MUSEUM = "музей"
    CHURCH = "церковь"
    MONASTERY = "монастырь"
    OBSERVATORY = "обсерватория"
    
    WATER_SOURCE = "источник_воды"
    FISHING_SPOT = "рыболовное_место"
    HUNTING_GROUND = "охотничьи_угодья"
    
    LANDMARK = "ориентир"
    UNKNOWN = "неизвестно"


class MapSourceType(Enum):
    
    YANDEX_MAPS = "yandex_maps"
    UNKNOWN = "unknown"



class MapSymbolRegistry:
    
    def __init__(self):
        logger.info("Initializing MapSymbolRegistry")
        
        self.color_terrain_mapping: Dict[str, StandardTerrainType] = {
            'dark_green': StandardTerrainType.FOREST,
            'light_green': StandardTerrainType.GRASSLAND,
            'yellow_green': StandardTerrainType.MEADOW,
            'brown': StandardTerrainType.ROCKS,
            'dark_brown': StandardTerrainType.SWAMP,
            'blue': StandardTerrainType.WATER,
            'light_blue': StandardTerrainType.WATER,
            'white': StandardTerrainType.GLACIER,
            'gray': StandardTerrainType.ROCKS,
            'tan': StandardTerrainType.SAND,
            'orange': StandardTerrainType.SETTLEMENT,
            'red': StandardTerrainType.ROAD_ASPHALT,
            'yellow': StandardTerrainType.SETTLEMENT,
        }
        
        logger.info(" MapSymbolRegistry Запуск")
    
    def osm_tag_to_terrain(self, key: str, value: str) -> Optional[StandardTerrainType]:
        return self.osm_terrain_mapping.get((key, value))
    
    def rgb_to_terrain(self, r: int, g: int, b: int, confidence: bool = False) -> Tuple[StandardTerrainType, float]:
        if g > r and g > b:
            if g > 200 and r < 100 and b < 100:
                terrain = StandardTerrainType.GRASSLAND
                conf = 0.85
            elif g > 150 and r > 80 and b < 100:
                terrain = StandardTerrainType.FOREST
                conf = 0.90
            else:
                terrain = StandardTerrainType.MEADOW
                conf = 0.75
        elif b > r and b > g:
            terrain = StandardTerrainType.WATER
            conf = 0.95
        elif r > 200 and g > 200 and b > 200:
            terrain = StandardTerrainType.GLACIER
            conf = 0.80
        elif r > g and r > b and g < 150 and b < 150:
            terrain = StandardTerrainType.ROAD_ASPHALT
            conf = 0.70
        elif r == g and g == b:
            terrain = StandardTerrainType.ROCKS
            conf = 0.75
        else:
            terrain = StandardTerrainType.UNKNOWN
            conf = 0.30
        
        if confidence:
            return (terrain, conf)
        return terrain



class UniversalSymbolDecoder:
    
    def __init__(self):
        self.registry = MapSymbolRegistry()
        logger.info("UniversalSymbolDecoder Запуск")
    
    def detect_map_source(self, html_or_path: str) -> MapSourceType:
        if '<svg class="ymaps-' in html_or_path or 'yandex' in html_or_path.lower():
            return MapSourceType.YANDEX_MAPS
        else:
            return MapSourceType.UNKNOWN
    
    def normalize_terrain_type(self, terrain: str, source: MapSourceType = MapSourceType.YANDEX_MAPS) -> Optional[StandardTerrainType]:
        terrain_lower = terrain.lower()
        for std_type in StandardTerrainType:
            if std_type.value == terrain_lower:
                return std_type
        
        fuzzy_map = {
            'тайга': StandardTerrainType.TAIGA,
            'лес': StandardTerrainType.FOREST,
            'болото': StandardTerrainType.SWAMP,
            'степь': StandardTerrainType.STEPPE,
            'тундра': StandardTerrainType.TUNDRA,
            'дорога': StandardTerrainType.ROAD_ASPHALT,
            'река': StandardTerrainType.RIVER,
            'озеро': StandardTerrainType.LAKE,
            'ледник': StandardTerrainType.GLACIER,
            'скалы': StandardTerrainType.ROCKS,
            'посёлок': StandardTerrainType.SETTLEMENT,
            'город': StandardTerrainType.TOWN,
        }
        
        return fuzzy_map.get(terrain_lower)



@dataclass
class TerrainAnalysisResult:
    
    terrain_type: StandardTerrainType
    confidence: float
    analysis_method: str
    color_components: Optional[Tuple[int, int, int]] = None
    texture_features: Optional[Dict] = None
    
    def __str__(self) -> str:
        return f"{self.terrain_type.value} (confidence: {self.confidence:.2f})"


class ColorAndTextureAnalyzer:
    
    def __init__(self):
        self.registry = MapSymbolRegistry()
        logger.info("ColorAndTextureAnalyzer Запуск")
    
    def analyze_pixel_color(self, r: int, g: int, b: int) -> TerrainAnalysisResult:
        terrain_type, confidence = self.registry.rgb_to_terrain(r, g, b, confidence=True)
        
        return TerrainAnalysisResult(
            terrain_type=terrain_type,
            confidence=confidence,
            analysis_method='color',
            color_components=(r, g, b)
        )
    
    def analyze_region_colors(self, pixels: List[Tuple[int, int, int]]) -> TerrainAnalysisResult:
        if not pixels:
            return TerrainAnalysisResult(
                terrain_type=StandardTerrainType.UNKNOWN,
                confidence=0.0,
                analysis_method='color'
            )
        
        terrain_votes = {}
        confidence_sum = 0.0
        
        for r, g, b in pixels:
            terrain, conf = self.registry.rgb_to_terrain(r, g, b, confidence=True)
            
            if terrain not in terrain_votes:
                terrain_votes[terrain] = 0
            
            terrain_votes[terrain] += conf
            confidence_sum += conf
        
        dominant_terrain = max(terrain_votes, key=terrain_votes.get)
        final_confidence = terrain_votes[dominant_terrain] / confidence_sum
        
        avg_r = sum(p[0] for p in pixels) // len(pixels)
        avg_g = sum(p[1] for p in pixels) // len(pixels)
        avg_b = sum(p[2] for p in pixels) // len(pixels)
        
        return TerrainAnalysisResult(
            terrain_type=dominant_terrain,
            confidence=final_confidence,
            analysis_method='color',
            color_components=(avg_r, avg_g, avg_b)
        )
    
    def estimate_texture_features(self, region_description: str) -> Dict[str, float]:
        features = {
            'smoothness': 0.5,
            'density': 0.5,
            'regularity': 0.5,
            'variation': 0.5,
        }
        
        description_lower = region_description.lower()
        
        if any(word in description_lower for word in ['лес', 'деревья', 'густо']):
            features['density'] = 0.9
            features['smoothness'] = 0.3
        
        if any(word in description_lower for word in ['поле', 'гладко', 'ровно']):
            features['smoothness'] = 0.8
            features['regularity'] = 0.7
        
        if any(word in description_lower for word in ['скалы', 'камни', 'скальный']):
            features['smoothness'] = 0.2
            features['variation'] = 0.8
        
        if any(word in description_lower for word in ['вода', 'озеро', 'река']):
            features['smoothness'] = 0.9
            features['regularity'] = 0.6
        
        return features



def initialize_decoder() -> UniversalSymbolDecoder:
    logger.info("=" * 80)
    logger.info(" Initializing universal Map decoder")
    logger.info("=" * 80)
    
    decoder = UniversalSymbolDecoder()
    
    logger.info(f" Supported standard terrain types: {len(StandardTerrainType)}")
    logger.info(f" Supported standard object types: {len(StandardObjectType)}")
    logger.info(f" OSM terrain mappings: {len(decoder.registry.osm_terrain_mapping)}")
    logger.info(f" OSM object mappings: {len(decoder.registry.osm_object_mapping)}")
    logger.info(" MapDecoder ready for use")
    
    return decoder


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    decoder = initialize_decoder()
    
    print("\n" + "=" * 80)
    print(" Testing OSM Tag Decoding")
    print("=" * 80)
    
    test_tags = {
        'landuse': 'forest',
        'natural': 'peak',
    }
    
    terrain, objects = decoder.decode_osm_tags(test_tags)
    print(f"Terrain: {terrain}")
    print(f"Objects: {objects}")
    
    print("\n" + "=" * 80)
    print(" Testing Color Analysis")
    print("=" * 80)
    
    analyzer = ColorAndTextureAnalyzer()
    
    test_colors = [
        (100, 150, 100, "Dark green (forest)"),
        (150, 200, 150, "Light green (meadow)"),
        (180, 180, 255, "Light blue (water)"),
        (200, 200, 200, "Gray (rocks)"),
    ]
    
    for r, g, b, description in test_colors:
        result = analyzer.analyze_pixel_color(r, g, b)
        print(f"{description}: {result}")

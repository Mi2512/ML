
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from map_decoder import StandardTerrainType, MapSymbolRegistry

logger = logging.getLogger(__name__)



@dataclass
class TerrainClassificationResult:
    
    terrain_type: StandardTerrainType
    confidence: float
    
    sources_used: List[str] = field(default_factory=list)
    source_votes: Dict[str, StandardTerrainType] = field(default_factory=dict)
    source_confidences: Dict[str, float] = field(default_factory=dict)
    
    total_votes: int = 0
    majority_votes: int = 0
    
    def __str__(self) -> str:
        conf_pct = int(self.confidence * 100)
        return f"{self.terrain_type.value} ({conf_pct}%)"
    
    def to_dict(self) -> dict:
        return {
            'terrain_type': self.terrain_type.value,
            'confidence': round(self.confidence, 2),
            'sources_used': self.sources_used,
            'source_confidences': {k: round(v, 2) for k, v in self.source_confidences.items()}
        }



class ConflictResolutionStrategy(Enum):
    
    YANDEX_PRIORITY = "yandex_priority"
    HIGHEST_CONFIDENCE = "highest_confidence"
    AVERAGE = "average"



class UniversalTerrainClassifier:
    
    def __init__(self, 
                 resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MAJORITY_VOTING):
        self.registry = MapSymbolRegistry()
        self.strategy = resolution_strategy
        
        logger.info(f"UniversalTerrainClassifier initialized (strategy: {resolution_strategy.value})")
    
    def classify_from_sources(self,
                            osm_terrain: Optional[StandardTerrainType] = None,
                            osm_confidence: float = 0.0,
                            csv_terrain: Optional[StandardTerrainType] = None,
                            satellite_terrain: Optional[StandardTerrainType] = None,
                            satellite_confidence: float = 0.0) -> TerrainClassificationResult:
        
        result = TerrainClassificationResult(
            terrain_type=StandardTerrainType.UNKNOWN,
            confidence=0.0
        )
        
        votes = {}
        confidences = {}
        
        if osm_terrain:
            votes['osm'] = osm_terrain
            confidences['osm'] = osm_confidence
            result.sources_used.append('osm')
            result.source_votes['osm'] = osm_terrain
            result.source_confidences['osm'] = osm_confidence
        
        if csv_terrain:
            votes['csv'] = csv_terrain
            confidences['csv'] = 0.9
            result.sources_used.append('csv')
            result.source_votes['csv'] = csv_terrain
            result.source_confidences['csv'] = 0.9
        
        if satellite_terrain:
            votes['satellite'] = satellite_terrain
            confidences['satellite'] = satellite_confidence
            result.sources_used.append('satellite')
            result.source_votes['satellite'] = satellite_terrain
            result.source_confidences['satellite'] = satellite_confidence
        
        result.total_votes = len(votes)
        
        if result.total_votes == 0:
            logger.warning("No terrain data from any source")
            return result
        elif result.total_votes == 1:
            only_source = list(votes.keys())[0]
            result.terrain_type = votes[only_source]
            result.confidence = confidences[only_source]
            result.resolution_method = "single_source"
        else:
            result = self._resolve_conflict(votes, confidences, result)
        
        return result
    
    def _resolve_conflict(self,
                         votes: Dict[str, StandardTerrainType],
                         confidences: Dict[str, float],
                         result: TerrainClassificationResult) -> TerrainClassificationResult:
        
        unique_types = set(votes.values())
        
        if len(unique_types) == 1:
            result.terrain_type = list(votes.values())[0]
            result.confidence = sum(confidences.values()) / len(confidences)
            result.conflict_detected = False
        else:
            result.conflict_detected = True
            
            if len(unique_types) == 2:
                type_counts = {}
                for t in votes.values():
                    type_counts[t] = type_counts.get(t, 0) + 1
                
                if any(count >= 2 for count in type_counts.values()):
                    result.conflict_type = "minor"
                else:
                    result.conflict_type = "major"
            else:
                result.conflict_type = "major"
            
            for source, terrain in votes.items():
                for other_source, other_terrain in votes.items():
                    if source != other_source and terrain != other_terrain:
                        if (source, terrain) not in result.conflicting_sources:
                            result.conflicting_sources.append((source, terrain))
            
            if self.strategy == ConflictResolutionStrategy.CSV_PRIORITY:
                result = self._resolve_csv_priority(votes, confidences, result)
            elif self.strategy == ConflictResolutionStrategy.OSM_PRIORITY:
                result = self._resolve_osm_priority(votes, confidences, result)
            elif self.strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
                result = self._resolve_highest_confidence(votes, confidences, result)
            elif self.strategy == ConflictResolutionStrategy.AVERAGE:
                result = self._resolve_average(votes, confidences, result)
            else:
                result = self._resolve_majority_voting(votes, confidences, result)
        
        return result
    
    def _resolve_majority_voting(self,
                                votes: Dict[str, StandardTerrainType],
                                confidences: Dict[str, float],
                                result: TerrainClassificationResult) -> TerrainClassificationResult:
        
        result.resolution_method = "majority_voting"
        
        type_votes = {}
        type_confidences = {}
        
        for source, terrain in votes.items():
            if terrain not in type_votes:
                type_votes[terrain] = 0
                type_confidences[terrain] = []
            
            type_votes[terrain] += 1
            type_confidences[terrain].append(confidences[source])
        
        max_votes = max(type_votes.values())
        result.majority_votes = max_votes
        
        if max_votes >= 2:
            dominant_terrain = max(type_votes, key=type_votes.get)
            result.terrain_type = dominant_terrain
            
            majority_confidences = type_confidences[dominant_terrain]
            result.confidence = sum(majority_confidences) / len(majority_confidences)
        else:
            best_source = max(confidences, key=confidences.get)
            result.terrain_type = votes[best_source]
            result.confidence = confidences[best_source]
        
        return result
    
    def _resolve_csv_priority(self,
                             votes: Dict[str, StandardTerrainType],
                             confidences: Dict[str, float],
                             result: TerrainClassificationResult) -> TerrainClassificationResult:
        
        result.resolution_method = "csv_priority"
        
        if 'csv' in votes:
            result.terrain_type = votes['csv']
            result.confidence = confidences['csv']
        else:
            return self._resolve_majority_voting(votes, confidences, result)
        
        return result
    
    def _resolve_osm_priority(self,
                             votes: Dict[str, StandardTerrainType],
                             confidences: Dict[str, float],
                             result: TerrainClassificationResult) -> TerrainClassificationResult:
        
        result.resolution_method = "osm_priority"
        
        if 'osm' in votes:
            result.terrain_type = votes['osm']
            result.confidence = confidences['osm']
        else:
            return self._resolve_majority_voting(votes, confidences, result)
        
        return result
    
    def _resolve_highest_confidence(self,
                                   votes: Dict[str, StandardTerrainType],
                                   confidences: Dict[str, float],
                                   result: TerrainClassificationResult) -> TerrainClassificationResult:
        
        result.resolution_method = "highest_confidence"
        
        best_source = max(confidences, key=confidences.get)
        result.terrain_type = votes[best_source]
        result.confidence = confidences[best_source]
        
        return result
    
    def _resolve_average(self,
                        votes: Dict[str, StandardTerrainType],
                        confidences: Dict[str, float],
                        result: TerrainClassificationResult) -> TerrainClassificationResult:
        
        result.resolution_method = "average"
        
        result = self._resolve_majority_voting(votes, confidences, result)
        
        avg_confidence = sum(confidences.values()) / len(confidences)
        result.confidence = avg_confidence
        
        return result



class ConflictDetectorAndLogger:
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.conflicts = []
        
        logger.info("ConflictDetectorAndLogger Запуск")
    
    def log_conflict(self,
                    track_id: str,
                    point_index: int,
                    latitude: float,
                    longitude: float,
                    csv_terrain: StandardTerrainType,
                    osm_terrain: StandardTerrainType,
                    satellite_terrain: Optional[StandardTerrainType] = None):
        
        conflict = {
            'track_id': track_id,
            'point_index': point_index,
            'latitude': latitude,
            'longitude': longitude,
            'csv_terrain': csv_terrain.value,
            'osm_terrain': osm_terrain.value,
            'satellite_terrain': satellite_terrain.value if satellite_terrain else None,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        self.conflicts.append(conflict)
        
        logger.warning(
            f"CONFLICT: {track_id}[{point_index}] at ({latitude:.4f}, {longitude:.4f}): "
            f"CSV='{csv_terrain.value}' vs OSM='{osm_terrain.value}'"
        )
        
        if self.log_file:
            self._write_to_file(conflict)
    
    def _write_to_file(self, conflict: dict):
        try:
            import json
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(conflict, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write conflict log: {e}")
    
    def get_conflict_statistics(self) -> dict:
        
        total = len(self.conflicts)
        
        if total == 0:
            return {
                'total_conflicts': 0,
                'conflict_percentage': 0.0,
                'conflict_locations': []
            }
        
        terrain_pairs = {}
        for conflict in self.conflicts:
            pair = (conflict['csv_terrain'], conflict['osm_terrain'])
            terrain_pairs[pair] = terrain_pairs.get(pair, 0) + 1
        
        return {
            'total_conflicts': total,
            'unique_conflict_types': len(terrain_pairs),
            'most_common_conflict': max(terrain_pairs, key=terrain_pairs.get) if terrain_pairs else None,
            'terrain_pair_distribution': terrain_pairs
        }



def initialize_terrain_analyzer(
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MAJORITY_VOTING,
    log_file: Optional[str] = None
) -> Tuple[UniversalTerrainClassifier, ConflictDetectorAndLogger]:
    
    logger.info("=" * 80)
    logger.info(" Initializing terrain ANALYZER")
    logger.info("=" * 80)
    
    classifier = UniversalTerrainClassifier(resolution_strategy)
    detector = ConflictDetectorAndLogger(log_file)
    
    logger.info(f" Terrain classifier initialized (strategy: {resolution_strategy.value})")
    logger.info(f" Conflict detector initialized")
    
    return (classifier, detector)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    classifier, detector = initialize_terrain_analyzer()
    
    print("\n" + "=" * 80)
    print(" Testing Terrain Classification")
    print("=" * 80)
    
    result = classifier.classify_from_sources(
        osm_terrain=StandardTerrainType.FOREST,
        osm_confidence=0.85,
        csv_terrain=StandardTerrainType.TAIGA,
        satellite_terrain=StandardTerrainType.FOREST,
        satellite_confidence=0.80
    )
    
    print(f"\nClassification result:")
    print(f"  Type: {result}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Sources: {result.sources_used}")
    print(f"  Conflict: {result.conflict_detected}")
    print(f"  Resolution: {result.resolution_method}")

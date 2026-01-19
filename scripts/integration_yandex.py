
import sys
import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_enricher_yandex import (
    initialize_enricher_yandex,
    EnrichedPointData,
    StandardTerrainType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage2YandexIntegrationTester:
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.engine = initialize_enricher_yandex()
        self.test_results = []
        
    def load_dataset(self, sample_size: int = 50) -> List[Dict]:
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found: {self.dataset_path}")
            return []
        
        points = []
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if sample_size > 0 and i >= sample_size:
                        break
                    
                    try:
                        point = {
                            'track_id': row.get('track_id', f'TRACK_{i//100}'),
                            'point_index': int(row.get('point_index', i)),
                            'latitude': float(row['latitude']),
                            'longitude': float(row['longitude']),
                            'altitude': float(row.get('altitude', 0)),
                            'date': row.get('date', '2024-01-17'),
                            'csv_terrain_type': self._parse_terrain(row.get('terrain_type'))
                        }
                        points.append(point)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Skipping row {i}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Dataset loading error: {e}")
        
        logger.info(f"Loaded {len(points)} points from dataset")
        return points
    
    def _parse_terrain(self, terrain_str: str) -> StandardTerrainType:
        if not terrain_str:
            return StandardTerrainType.GRASSLAND
        
        terrain_str = terrain_str.lower()
        
        for terrain_type in StandardTerrainType:
            if terrain_type.value.lower() in terrain_str:
                return terrain_type
        
        return StandardTerrainType.GRASSLAND
    
    def run_test(self, points: List[Dict], test_name: str = "Yandex Integration Test"):
        logger.info(f"\n{'='*80}")
        logger.info(f"  {test_name}")
        logger.info(f"{'='*80}\n")
        
        if not points:
            logger.error("No test points provided")
            return None
        
        logger.info(f"Processing {len(points)} points...")
        start_time = time.time()
        
        enriched_points = self.engine.enrich_batch(points)
        
        elapsed = time.time() - start_time
        
        metrics = self._calculate_metrics(enriched_points, elapsed)
        
        validation = self._validate_results(enriched_points)
        
        self._print_results(enriched_points, metrics, validation, elapsed)
        
        self._save_results(enriched_points, metrics)
        
        return {
            'enriched_points': enriched_points,
            'metrics': metrics,
            'validation': validation
        }
    
    def _calculate_metrics(self, enriched_points: List[EnrichedPointData], elapsed: float) -> Dict:
        if not enriched_points:
            return {}
        
        metrics = {
            'total_points': len(enriched_points),
            'total_time_s': round(elapsed, 2),
            'avg_time_per_point_ms': round(elapsed / len(enriched_points) * 1000, 2),
            'points_per_second': round(len(enriched_points) / elapsed, 2),
            'success_rate': f"{sum(1 for p in enriched_points if p.yandex_available) / len(enriched_points) * 100:.1f}%",
        }
        
        terrain_coverage = sum(1 for p in enriched_points if p.terrain_type)
        metrics['terrain_coverage'] = f"{terrain_coverage / len(enriched_points) * 100:.1f}%"
        
        avg_objects = sum(p.object_count for p in enriched_points) / len(enriched_points)
        metrics['avg_objects_per_point'] = round(avg_objects, 2)
        
        avg_quality = sum(p.quality_score for p in enriched_points) / len(enriched_points)
        metrics['avg_quality_score'] = round(avg_quality, 3)
        
        return metrics
    
    def _validate_results(self, enriched_points: List[EnrichedPointData]) -> Dict:
        validation = {
            'total_points': len(enriched_points),
            'points_with_terrain': sum(1 for p in enriched_points if p.terrain_type),
            'points_with_objects': sum(1 for p in enriched_points if p.object_count > 0),
            'avg_objects': round(
                sum(p.object_count for p in enriched_points) / len(enriched_points), 2
            ) if enriched_points else 0,
            'failed_enrichments': sum(1 for p in enriched_points if not p.yandex_available),
            'yandex_available_rate': f"{sum(1 for p in enriched_points if p.yandex_available) / len(enriched_points) * 100:.1f}%" if enriched_points else "0%"
        }
        
        return validation
    
    def _print_results(self, enriched_points: List[EnrichedPointData], 
                      metrics: Dict, validation: Dict, elapsed: float):
        print("\n" + "="*80)
        print("Integration test RESULTS")
        print("="*80 + "\n")
        
        print("Performance metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"  {key:30} : {value}")
        
        print("\n\nValidation results:")
        print("-" * 40)
        for key, value in validation.items():
            print(f"  {key:30} : {value}")
        
        print("\n\nSample enriched POINTS:")
        print("-" * 40)
        
        for i, point in enumerate(enriched_points[:5]):
            print(f"\nPoint {i+1}:")
            print(f"  Location: ({point.latitude:.4f}, {point.longitude:.4f})")
            print(f"  Terrain: {point.terrain_type}")
            print(f"  Terrain confidence: {point.terrain_confidence:.2f}")
            print(f"  Nearby objects: {point.object_count}")
            print(f"  Quality score: {point.quality_score:.2f}")
            print(f"  Yandex available: {point.yandex_available}")
        
        print("\n" + "="*80 + "\n")
    
    def _save_results(self, enriched_points: List[EnrichedPointData], metrics: Dict):
        output_file = Path("stage2_yandex_test_results.json")
        
        try:
            results = {
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics,
                'sample_points': [p.to_dict() for p in enriched_points[:10]],
                'total_enriched': len(enriched_points)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    
    dataset_path = Path(__file__).parent.parent / "dataset" / "large_route_dataset_20260117_143350.csv"
    
    if not dataset_path.exists():
        dataset_path = Path("dataset/large_route_dataset_20260117_143350.csv")
    
    print("\n" + "="*80)
    print("Integration test - Yandex version")
    print("="*80 + "\n")
    
    tester = Stage2YandexIntegrationTester(str(dataset_path))
    
    print("Loading sample data")
    test_points = tester.load_dataset(sample_size=50)
    
    if test_points:
        print(f"\n Loaded {len(test_points)} test points")
        result = tester.run_test(test_points, "Yandex API Integration Test (50 points)")
        
        if result:
            print("\n Integration test Готово!")
    else:
        print("\n Ошибка test data")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

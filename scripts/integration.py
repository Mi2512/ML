
import sys
import os
from pathlib import Path
import pandas as pd
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from map_enricher_optimized import (
    EnrichmentEngine,
    initialize_enricher
)
from map_decoder import StandardTerrainType, StandardObjectType


class Stage2IntegrationTester:
    
    def __init__(self, dataset_path: str, sample_size: int = 50):
        self.dataset_path = Path(dataset_path)
        self.sample_size = sample_size
        self.engine = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'sample_size': sample_size,
            'metrics': {},
            'errors': []
        }
    
    def load_dataset(self) -> pd.DataFrame:
        print(f"\n Loading dataset from: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        print(f"    Loaded {len(df)} total points")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
        
        return df
    
    def prepare_test_data(self, df: pd.DataFrame) -> list:
        print(f"\n Preparing test data (sample size: {self.sample_size})")
        
        step = max(1, len(df) // self.sample_size)
        sampled_df = df.iloc[::step].head(self.sample_size)
        
        test_points = []
        for idx, row in sampled_df.iterrows():
            try:
                point = {
                    'track_id': str(row.get('track_id', f"TRACK_{idx}")),
                    'point_index': int(idx),
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'altitude': float(row.get('altitude', 0.0)),
                    'date': str(row.get('date', '2024-01-17')),
                    'csv_terrain_type': None
                }
                
                if 'terrain_type' in row and pd.notna(row['terrain_type']):
                    try:
                        terrain_str = str(row['terrain_type']).lower()
                        terrain_map = {
                            'forest': StandardTerrainType.FOREST,
                            'grassland': StandardTerrainType.GRASSLAND,
                            'water': StandardTerrainType.WATER,
                            'swamp': StandardTerrainType.SWAMP,
                            'tundra': StandardTerrainType.TUNDRA,
                            'desert': StandardTerrainType.DESERT
                        }
                        for key, terrain in terrain_map.items():
                            if key in terrain_str:
                                point['csv_terrain_type'] = terrain
                                break
                    except:
                        pass
                
                test_points.append(point)
            except Exception as e:
                print(f"     Error preparing point {idx}: {e}")
                continue
        
        print(f"    Prepared {len(test_points)} test points")
        return test_points
    
    def run_enrichment_tests(self, test_points: list):
        print(f"\n Running enrichment tests on {len(test_points)} points...")
        
        self.engine = initialize_enricher(cache_dir="osm_cache_stage2")
        
        start_time = time.time()
        enriched_points = []
        
        for i, point in enumerate(test_points):
            try:
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(test_points) - i - 1) / rate if rate > 0 else 0
                    print(f"   Progress: {i+1}/{len(test_points)} ({rate:.2f} pts/sec, ETA: {remaining:.0f}s)")
                
                enriched = self.engine.enrich_point(
                    track_id=point['track_id'],
                    point_index=point['point_index'],
                    latitude=point['latitude'],
                    longitude=point['longitude'],
                    altitude=point['altitude'],
                    date=point['date'],
                    csv_terrain_type=point['csv_terrain_type']
                )
                
                enriched_points.append(enriched)
                
            except Exception as e:
                self.results['errors'].append({
                    'point_index': point['point_index'],
                    'error': str(e)
                })
                print(f"    Error enriching point {point['point_index']}: {e}")
        
        elapsed = time.time() - start_time
        
        self.results['metrics']['total_time_seconds'] = round(elapsed, 2)
        self.results['metrics']['points_processed'] = len(enriched_points)
        self.results['metrics']['points_failed'] = len(test_points) - len(enriched_points)
        self.results['metrics']['avg_time_per_point'] = round(elapsed / len(enriched_points), 2) if enriched_points else 0
        self.results['metrics']['points_per_second'] = round(len(enriched_points) / elapsed, 2) if elapsed > 0 else 0
        
        print(f"    Enrichment complete!")
        print(f"     - Time elapsed: {elapsed:.1f}s")
        print(f"     - Points processed: {len(enriched_points)}")
        print(f"     - Avg time/point: {self.results['metrics']['avg_time_per_point']:.2f}s")
        print(f"     - Rate: {self.results['metrics']['points_per_second']:.2f} pts/sec")
        
        return enriched_points
    
    def validate_results(self, enriched_points: list):
        print(f"\n Validating results...")
        
        terrain_coverage = sum(1 for p in enriched_points if p.terrain_type)
        objects_found = sum(1 for p in enriched_points if p.object_count > 0)
        avg_quality = sum(p.quality_score for p in enriched_points) / len(enriched_points)
        
        self.results['metrics']['terrain_coverage'] = f"{terrain_coverage}/{len(enriched_points)} ({terrain_coverage/len(enriched_points)*100:.1f}%)"
        self.results['metrics']['points_with_objects'] = f"{objects_found}/{len(enriched_points)} ({objects_found/len(enriched_points)*100:.1f}%)"
        self.results['metrics']['avg_quality_score'] = round(avg_quality, 2)
        self.results['metrics']['total_objects_found'] = sum(p.object_count for p in enriched_points)
        self.results['metrics']['avg_objects_per_point'] = round(sum(p.object_count for p in enriched_points) / len(enriched_points), 1)
        
        print(f"    Terrain coverage: {terrain_coverage}/{len(enriched_points)} ({terrain_coverage/len(enriched_points)*100:.1f}%)")
        print(f"    Points with objects: {objects_found}/{len(enriched_points)} ({objects_found/len(enriched_points)*100:.1f}%)")
        print(f"    Total objects found: {self.results['metrics']['total_objects_found']}")
        print(f"    Avg objects per point: {self.results['metrics']['avg_objects_per_point']}")
        print(f"    Avg quality score: {avg_quality:.2f}")
    
    def check_optimization_metrics(self):
        print(f"\n Checking optimization metrics...")
        
        if self.engine:
            stats = self.engine.extractor.get_statistics()
            
            cache_stats = stats.get('cache_stats', {})
            self.results['metrics']['cache_hits'] = cache_stats.get('hits', 0)
            self.results['metrics']['cache_misses'] = cache_stats.get('misses', 0)
            self.results['metrics']['cache_hit_rate'] = cache_stats.get('hit_rate', '0%')
            self.results['metrics']['cached_files'] = cache_stats.get('cached_files', 0)
            
            limiter_stats = stats.get('rate_limiter_stats', {})
            self.results['metrics']['total_api_requests'] = limiter_stats.get('total_requests', 0)
            self.results['metrics']['rate_limiter_min_interval'] = limiter_stats.get('min_interval', '0s')
            
            print(f"    Cache hits: {cache_stats.get('hits', 0)}")
            print(f"    Cache misses: {cache_stats.get('misses', 0)}")
            print(f"    Cache hit rate: {cache_stats.get('hit_rate', '0%')}")
            print(f"    Cached files: {cache_stats.get('cached_files', 0)}")
            print(f"    Total API requests: {limiter_stats.get('total_requests', 0)}")
    
    def save_results(self, output_path: str = "results/stage2_integration_results.json"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n Results saved to: {output_path}")
        return output_path

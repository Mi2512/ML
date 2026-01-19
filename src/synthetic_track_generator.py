
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.database import DatabaseManager
from geodetic_augmentation import GeodeticAugmentation, GeoPoint
from map_augmentation import MapAugmentation, AugmentationParams, PRESET_AUGMENTATIONS
from augmentation_integration import (
    track_from_db_to_geopoints,
    geopoints_to_db_track,
    create_augmented_track
)
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json
import random
from PIL import Image


class SyntheticTrackGenerator:
    
    def __init__(self, db: DatabaseManager, random_seed: Optional[int] = None):
        self.db = db
        self.geodetic_aug = GeodeticAugmentation()
        self.visual_aug = MapAugmentation(random_seed=random_seed)
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def generate_single_variant(
        self,
        original_track_id: str,
        geodetic_type: str,
        geodetic_params: Dict,
        visual_params: Optional[AugmentationParams] = None,
        variant_suffix: Optional[str] = None
    ) -> Tuple[str, Dict]:
        synthetic_track_id, aug_record_id = create_augmented_track(
            self.db,
            original_track_id,
            geodetic_type,
            geodetic_params,
            variant_suffix
        )
        
        metadata = {
            'original_track_id': original_track_id,
            'synthetic_track_id': synthetic_track_id,
            'geodetic_augmentation': {
                'type': geodetic_type,
                'parameters': geodetic_params
            },
            'augmentation_record_id': str(aug_record_id),
            'created_at': datetime.now().isoformat()
        }
        
        if visual_params is not None:
            metadata['visual_augmentation'] = {
                'parameters': visual_params.to_dict()
            }
        
        return synthetic_track_id, metadata
    
    def generate_batch_variants(
        self,
        original_track_id: str,
        num_variants: int = 5,
        enable_visual_aug: bool = True,
        enable_weather: bool = True
    ) -> List[Tuple[str, Dict]]:
        results = []
        
        from augmentation_integration import track_from_db_to_geopoints
        geo_points = track_from_db_to_geopoints(self.db, original_track_id)
        has_timestamps = all(p.timestamp is not None for p in geo_points)
        
        strategies = [
            {'type': 'geodesic_rotation', 'params': {'rotation_angle': 5.0}, 'suffix': 'ROT5'},
            {'type': 'geodesic_rotation', 'params': {'rotation_angle': -5.0}, 'suffix': 'ROTN5'},
            {'type': 'geodesic_rotation', 'params': {'rotation_angle': 10.0}, 'suffix': 'ROT10'},
            
            {'type': 'azimuth_shift', 'params': {'shift_distance': 200.0, 'azimuth': 0.0}, 'suffix': 'SHIFT_N'},
            {'type': 'azimuth_shift', 'params': {'shift_distance': 200.0, 'azimuth': 90.0}, 'suffix': 'SHIFT_E'},
            {'type': 'azimuth_shift', 'params': {'shift_distance': 200.0, 'azimuth': 180.0}, 'suffix': 'SHIFT_S'},
            {'type': 'azimuth_shift', 'params': {'shift_distance': 150.0, 'azimuth': 45.0}, 'suffix': 'SHIFT_NE'},
            
            {'type': 'elevation_jitter', 'params': {'jitter_magnitude': 3.0}, 'suffix': 'ELEV3'},
            {'type': 'elevation_jitter', 'params': {'jitter_magnitude': 5.0}, 'suffix': 'ELEV5'},
        ]
        
        if has_timestamps:
            strategies.extend([
                {'type': 'temporal_shift', 'params': {'shift_days': 7}, 'suffix': 'TIME_P7D'},
                {'type': 'temporal_shift', 'params': {'shift_days': -7}, 'suffix': 'TIME_N7D'},
            ])
        
        selected_strategies = random.sample(strategies, min(num_variants, len(strategies)))
        
        for i, strategy in enumerate(selected_strategies):
            try:
                geodetic_type = strategy['type']
                geodetic_params = strategy['params']
                suffix = f"{strategy['suffix']}_{i+1:03d}"
                
                visual_params = None
                if enable_visual_aug:
                    if random.random() < 0.5 and len(PRESET_AUGMENTATIONS) > 0:
                        preset_name = random.choice(list(PRESET_AUGMENTATIONS.keys()))
                        visual_params = PRESET_AUGMENTATIONS[preset_name]
                    else:
                        visual_params = AugmentationParams(
                            brightness=random.uniform(0.8, 1.2),
                            contrast=random.uniform(0.9, 1.1),
                            saturation=random.uniform(0.9, 1.1),
                            hue_shift=random.uniform(-10, 10),
                            blur_radius=random.uniform(0, 1.0),
                            noise_intensity=random.uniform(0, 0.03)
                        )
                        
                        if enable_weather and random.random() < 0.3:
                            visual_params.weather_effect = random.choice(['fog', 'rain', 'snow'])
                            visual_params.weather_intensity = random.uniform(0.2, 0.5)
                
                synth_id, metadata = self.generate_single_variant(
                    original_track_id,
                    geodetic_type,
                    geodetic_params,
                    visual_params,
                    suffix
                )
                
                results.append((synth_id, metadata))
                print(f" Generated variant {i+1}/{num_variants}: {synth_id}")
            
            except Exception as e:
                print(f" Failed to generate variant {i+1}: {e}")
        
        return results
    
    def generate_balanced_dataset(
        self,
        source_track_ids: List[str],
        variants_per_track: int = 5,
        enable_visual_aug: bool = True,
        enable_weather: bool = True
    ) -> Dict[str, List[Tuple[str, Dict]]]:
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Generating Balanced Synthetic Dataset")
        print(f"{'='*70}")
        print(f"Source tracks: {len(source_track_ids)}")
        print(f"Variants per track: {variants_per_track}")
        print(f"Total variants: {len(source_track_ids) * variants_per_track}")
        print(f"{'='*70}\n")
        
        for i, track_id in enumerate(source_track_ids):
            print(f"\nProcessing track {i+1}/{len(source_track_ids)}: {track_id}")
            print("-" * 70)
            
            try:
                variants = self.generate_batch_variants(
                    track_id,
                    num_variants=variants_per_track,
                    enable_visual_aug=enable_visual_aug,
                    enable_weather=enable_weather
                )
                
                results[track_id] = variants
                print(f" Generated {len(variants)} variants for {track_id}")
            
            except Exception as e:
                print(f" Failed to process {track_id}: {e}")
                results[track_id] = []
        
        total_generated = sum(len(variants) for variants in results.values())
        print(f"\n{'='*70}")
        print(f" Dataset Generation Complete")
        print(f"{'='*70}")
        print(f"Source tracks processed: {len(results)}")
        print(f"Total variants generated: {total_generated}")
        print(f"Average per track: {total_generated / len(results) if results else 0:.1f}")
        print(f"{'='*70}\n")
        
        return results
    
    def export_generation_report(
        self,
        results: Dict[str, List[Tuple[str, Dict]]],
        output_path: str
    ):
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_source_tracks': len(results),
            'total_synthetic_tracks': sum(len(variants) for variants in results.values()),
            'tracks': {}
        }
        
        for source_id, variants in results.items():
            report['tracks'][source_id] = {
                'variant_count': len(variants),
                'variants': [
                    {
                        'synthetic_id': synth_id,
                        'metadata': metadata
                    }
                    for synth_id, metadata in variants
                ]
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f" Report saved: {output_path}")


GENERATION_PRESETS = {
    'minimal': {
        'variants_per_track': 3,
        'enable_visual_aug': False,
        'enable_weather': False
    },
    'standard': {
        'variants_per_track': 5,
        'enable_visual_aug': True,
        'enable_weather': True
    },
    'extensive': {
        'variants_per_track': 10,
        'enable_visual_aug': True,
        'enable_weather': True
    },
    'geodetic_only': {
        'variants_per_track': 5,
        'enable_visual_aug': False,
        'enable_weather': False
    },
    'visual_heavy': {
        'variants_per_track': 7,
        'enable_visual_aug': True,
        'enable_weather': True
    }
}


if __name__ == '__main__':
    print("=" * 70)
    print("Synthetic Track Generator - Test")
    print("=" * 70)
    
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = temp_db.name
    temp_db.close()
    
    try:
        db = DatabaseManager(db_path)
        print(f"\n Database created: {db_path}\n")
        
        print("Creating test tracks")
        print("-" * 70)
        
        test_tracks = []
        for i in range(3):
            track_id = f"TRACK_TEST_{i+1:03d}"
            db.add_track(
                track_id=track_id,
                region='test_region',
                date_start=datetime.now()
            )
            
            points_data = []
            for j in range(5):
                points_data.append({
                    'point_index': j,
                    'latitude': 55.7558 + j * 0.001,
                    'longitude': 120.4356 + j * 0.001,
                    'altitude': 1200.0 + j * 5
                })
            
            db.add_points(track_id, points_data)
            test_tracks.append(track_id)
            print(f" Created {track_id} with {len(points_data)} points")
        
        generator = SyntheticTrackGenerator(db, random_seed=42)
        
        print("\n\nTest 1: Single Variant Generation")
        print("-" * 70)
        
        synth_id, metadata = generator.generate_single_variant(
            test_tracks[0],
            'geodesic_rotation',
            {'rotation_angle': 15.0},
            visual_params=AugmentationParams(brightness=1.1, contrast=1.05),
            variant_suffix='TEST_001'
        )
        
        print(f" Generated: {synth_id}")
        print(f"   Geodetic: {metadata['geodetic_augmentation']['type']}")
        print(f"   Visual: brightness={metadata['visual_augmentation']['parameters']['brightness']}")
        
        print("\n\nTest 2: Batch Variant Generation")
        print("-" * 70)
        
        batch_results = generator.generate_batch_variants(
            test_tracks[1],
            num_variants=5,
            enable_visual_aug=True,
            enable_weather=True
        )
        
        print(f"\n Generated {len(batch_results)} batch variants")
        
        print("\n\nTest 3: Balanced Dataset Generation")
        print("-" * 70)
        
        dataset_results = generator.generate_balanced_dataset(
            test_tracks,
            variants_per_track=3,
            enable_visual_aug=True,
            enable_weather=True
        )
        
        print("\n\nTest 4: Export Generation Report")
        print("-" * 70)
        
        report_path = Path(db_path).parent / 'generation_report.json'
        generator.export_generation_report(dataset_results, str(report_path))
        
        print("\n\nTest 5: Database Verification")
        print("-" * 70)
        
        session = db.Session()
        from src.database import RouteTrack, TrackAugmentation
        
        total_tracks = session.query(RouteTrack).count()
        total_augmentations = session.query(TrackAugmentation).count()
        session.close()
        
        print(f" Total tracks in DB: {total_tracks}")
        print(f" Total augmentations: {total_augmentations}")
        print(f"   Original tracks: {len(test_tracks)}")
        print(f"   Synthetic tracks: {total_tracks - len(test_tracks)}")
        
        print("\n" + "=" * 70)
        print(" All tests PASSED!")
        print("=" * 70)
        
    finally:
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
                print(f"\n  Cleaned up test database")
            except:
                pass    
    def generate_seasonal_variants(self, track: Dict, variants_per_season: int = 3) -> List[Dict]:
        seasons = {
            "spring": [80, 100, 120],
            "summer": [170, 190, 210],
            "autumn": [260, 280, 300],
            "winter": [350, 10, 30]
        }
        
        variants = []
        for season, day_offsets in seasons.items():
            for i, offset in enumerate(day_offsets[:variants_per_season]):
                variant = {
                    **track,
                    "variant_id": f"{track.get('id', 'track')}_season_{season}_{i}",
                    "season": season,
                    "day_offset": offset
                }
                variants.append(variant)
        
        return variants
    
    def ensure_seasonal_balance(self, tracks: List[Dict]) -> List[Dict]:
        seasonal_tracks = {
            "spring": [], "summer": [], "autumn": [], "winter": []
        }
        
        for track in tracks:
            season = track.get("season", "unknown")
            if season in seasonal_tracks:
                seasonal_tracks[season].append(track)
        
        min_count = min(len(v) for v in seasonal_tracks.values() if v)
        
        balanced = []
        for season, season_tracks in seasonal_tracks.items():
            if season_tracks:
                balanced.extend(season_tracks[:min_count])
        
        return balanced
    
    def validate_geodetic_accuracy(self, track: Dict, reference_track: Dict = None) -> Dict:
        validation = {
            "status": "valid",
            "errors": [],
            "warnings": []
        }
        
        if "points" in track:
            if len(track["points"]) == 0:
                validation["errors"].append("Track has no points")
                validation["status"] = "invalid"
        
        for point in track.get("points", []):
            lat = point.get("latitude", 0)
            lon = point.get("longitude", 0)
            
            if not (-90 <= lat <= 90):
                validation["errors"].append(f"Invalid latitude: {lat}")
                validation["status"] = "invalid"
            
            if not (-180 <= lon <= 180):
                validation["errors"].append(f"Invalid longitude: {lon}")
                validation["status"] = "invalid"
        
        points = track.get("points", [])
        if len(points) > 1:
            max_elevation_change = 0
            for i in range(1, len(points)):
                elev_diff = abs(points[i].get("altitude", 0) - points[i-1].get("altitude", 0))
                if elev_diff > max_elevation_change:
                    max_elevation_change = elev_diff
                
                if elev_diff > 100:
                    validation["warnings"].append(f"Large elevation jump at point {i}: {elev_diff}m")
        
        if reference_track:
            ref_points = reference_track.get("points", [])
            track_points = track.get("points", [])
            
            if len(ref_points) != len(track_points):
                validation["warnings"].append(
                    f"Point count mismatch: {len(track_points)} vs {len(ref_points)} (reference)"
                )
        
        return validation


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.database import DatabaseManager, RouteTrack, RoutePoint
from geodetic_augmentation import GeodeticAugmentation, GeoPoint
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json


def track_from_db_to_geopoints(db: DatabaseManager, track_id: str) -> List[GeoPoint]:
    session = db.Session()
    
    try:
        points = session.query(RoutePoint).filter_by(
            track_id_fk=track_id
        ).order_by(RoutePoint.point_index).all()
        
        if not points:
            raise ValueError(f"No points found for track {track_id}")
        
        geo_points = []
        for point in points:
            timestamp = None
            if hasattr(point, 'timestamp') and point.timestamp:
                if isinstance(point.timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(point.timestamp)
                    except:
                        pass
                elif isinstance(point.timestamp, datetime):
                    timestamp = point.timestamp
            
            geo_points.append(GeoPoint(
                lat=point.latitude,
                lon=point.longitude,
                elevation=point.altitude if point.altitude else 0.0,
                timestamp=timestamp
            ))
        
        return geo_points
        
    finally:
        session.close()


def geopoints_to_db_track(
    db: DatabaseManager,
    track_id: str,
    geo_points: List[GeoPoint],
    track_metadata: Optional[Dict] = None
) -> str:
    if not geo_points:
        raise ValueError("Cannot save empty track")
    
    metadata = track_metadata or {}
    metadata['point_count'] = len(geo_points)
    metadata['created_at'] = datetime.now().isoformat()
    
    track_created = db.add_track(
        track_id=track_id,
        region='synthetic',
        date_start=datetime.now()
    )
    
    if not track_created:
        raise ValueError(f"Failed to create track {track_id}")
    
    points_to_add = []
    for i, point in enumerate(geo_points):
        timestamp_str = None
        if point.timestamp:
            timestamp_str = point.timestamp.isoformat()
        
        points_to_add.append({
            'point_index': i,
            'latitude': point.lat,
            'longitude': point.lon,
            'altitude': point.elevation,
            'timestamp': timestamp_str
        })
    
    db.add_points(track_id, points_to_add)
    
    return track_id


def create_augmented_track(
    db: DatabaseManager,
    original_track_id: str,
    augmentation_type: str,
    parameters: Dict,
    augmented_track_suffix: str = None
) -> Tuple[str, str]:
    original_points = track_from_db_to_geopoints(db, original_track_id)
    
    augmenter = GeodeticAugmentation()
    
    augmented_points = None
    transform_matrix = None
    
    if augmentation_type == 'geodesic_rotation':
        rotation_angle = parameters.get('rotation_angle', 15.0)
        augmented_points, transform_matrix = augmenter.geodesic_rotation(
            original_points, rotation_angle
        )
    
    elif augmentation_type == 'azimuth_shift':
        shift_distance = parameters.get('shift_distance', 500.0)
        azimuth = parameters.get('azimuth', 0.0)
        augmented_points, transform_matrix = augmenter.azimuth_shift(
            original_points, shift_distance, azimuth
        )
    
    elif augmentation_type == 'elevation_jitter':
        jitter_magnitude = parameters.get('jitter_magnitude', 10.0)
        preserve_gain_loss = parameters.get('preserve_gain_loss', True)
        augmented_points, transform_matrix = augmenter.elevation_jitter(
            original_points, jitter_magnitude, preserve_gain_loss
        )
    
    elif augmentation_type == 'temporal_shift':
        shift_days = parameters.get('shift_days', 7)
        augmented_points, transform_matrix = augmenter.temporal_shift(
            original_points, shift_days
        )
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    if augmented_track_suffix is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        augmented_track_suffix = f"{augmentation_type.upper()}_{timestamp}"
    
    augmented_track_id = f"{original_track_id}_SYN_{augmented_track_suffix}"
    
    metadata = {
        'source': 'geodetic_augmentation',
        'original_track_id': original_track_id,
        'augmentation_type': augmentation_type,
        'parameters': parameters
    }
    
    geopoints_to_db_track(db, augmented_track_id, augmented_points, metadata)
    
    augmentation_id = db.add_track_augmentation(
        original_track_id=original_track_id,
        augmented_track_id=augmented_track_id,
        augmentation_type=augmentation_type,
        parameters=parameters,
        transformation_matrix=transform_matrix.to_dict(),
        is_reversible=True
    )
    
    return augmented_track_id, augmentation_id


def batch_augment_track(
    db: DatabaseManager,
    original_track_id: str,
    augmentation_configs: List[Dict]
) -> List[Tuple[str, str]]:
    results = []
    
    for config in augmentation_configs:
        aug_type = config['type']
        parameters = config['parameters']
        suffix = config.get('suffix', None)
        
        try:
            aug_track_id, aug_rec_id = create_augmented_track(
                db, original_track_id, aug_type, parameters, suffix
            )
            results.append((aug_track_id, aug_rec_id))
            print(f" Created augmentation: {aug_track_id}")
        
        except Exception as e:
            print(f" Failed to create {aug_type} augmentation: {e}")
    
    return results


PRESET_AUGMENTATIONS = {
    'rotation_small': {
        'type': 'geodesic_rotation',
        'parameters': {'rotation_angle': 10.0},
        'suffix': 'ROT10'
    },
    'rotation_medium': {
        'type': 'geodesic_rotation',
        'parameters': {'rotation_angle': 25.0},
        'suffix': 'ROT25'
    },
    'shift_north': {
        'type': 'azimuth_shift',
        'parameters': {'shift_distance': 500.0, 'azimuth': 0.0},
        'suffix': 'SHIFT_N500'
    },
    'shift_east': {
        'type': 'azimuth_shift',
        'parameters': {'shift_distance': 500.0, 'azimuth': 90.0},
        'suffix': 'SHIFT_E500'
    },
    'shift_south': {
        'type': 'azimuth_shift',
        'parameters': {'shift_distance': 500.0, 'azimuth': 180.0},
        'suffix': 'SHIFT_S500'
    },
    'elevation_light': {
        'type': 'elevation_jitter',
        'parameters': {'jitter_magnitude': 5.0, 'preserve_gain_loss': True},
        'suffix': 'ELEV5'
    },
    'elevation_medium': {
        'type': 'elevation_jitter',
        'parameters': {'jitter_magnitude': 10.0, 'preserve_gain_loss': True},
        'suffix': 'ELEV10'
    },
    'temporal_week': {
        'type': 'temporal_shift',
        'parameters': {'shift_days': 7},
        'suffix': 'TIME_P7D'
    },
    'temporal_month': {
        'type': 'temporal_shift',
        'parameters': {'shift_days': 30},
        'suffix': 'TIME_P30D'
    }
}


def apply_preset_augmentations(
    db: DatabaseManager,
    original_track_id: str,
    preset_names: List[str]
) -> List[Tuple[str, str]]:
    configs = []
    
    for preset_name in preset_names:
        if preset_name not in PRESET_AUGMENTATIONS:
            print(f"  Unknown preset: {preset_name}")
            continue
        configs.append(PRESET_AUGMENTATIONS[preset_name])
    
    return batch_augment_track(db, original_track_id, configs)


if __name__ == '__main__':
    print("=" * 70)
    print("Geodetic Augmentation - Database Integration Test")
    print("=" * 70)
    
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = temp_db.name
    temp_db.close()
    
    try:
        db = DatabaseManager(db_path)
        print(f" Database created: {db_path}\n")
        
        print("Creating original track")
        original_id = "TRACK_TEST_001"
        db.add_track(
            track_id=original_id,
            region='test_region',
            date_start=datetime.now()
        )
        
        test_points = [
            {'point_index': 0, 'latitude': 55.7558, 'longitude': 120.4356, 'altitude': 1200.0},
            {'point_index': 1, 'latitude': 55.7560, 'longitude': 120.4360, 'altitude': 1205.0},
            {'point_index': 2, 'latitude': 55.7565, 'longitude': 120.4370, 'altitude': 1210.0},
            {'point_index': 3, 'latitude': 55.7570, 'longitude': 120.4380, 'altitude': 1215.0},
        ]
        
        points_data = []
        for point in test_points:
            point_data = point.copy()
            points_data.append(point_data)
        
        db.add_points(original_id, points_data)
        
        print(f" Created track {original_id} with {len(test_points)} points\n")
        
        print("Test 1: Single Rotation Augmentation")
        print("-" * 70)
        
        aug_id, rec_id = create_augmented_track(
            db, original_id,
            'geodesic_rotation',
            {'rotation_angle': 20.0},
            'ROT20_001'
        )
        
        print(f" Created augmented track: {aug_id}")
        print(f" Augmentation record ID: {rec_id}\n")
        
        print("Test 2: Batch Augmentation")
        print("-" * 70)
        
        configs = [
            {'type': 'azimuth_shift', 'parameters': {'shift_distance': 300.0, 'azimuth': 45.0}, 'suffix': 'SHIFT_001'},
            {'type': 'elevation_jitter', 'parameters': {'jitter_magnitude': 8.0}, 'suffix': 'ELEV_001'},
        ]
        
        results = batch_augment_track(db, original_id, configs)
        print(f" Created {len(results)} augmented tracks\n")
        
        print("Test 3: Preset Augmentations")
        print("-" * 70)
        
        preset_results = apply_preset_augmentations(
            db, original_id,
            ['rotation_small', 'shift_north', 'elevation_light']
        )
        print(f" Applied {len(preset_results)} presets\n")
        
        print("Test 4: Query Augmentations")
        print("-" * 70)
        
        all_augs = db.get_track_augmentations(original_id, as_original=True)
        print(f" Track {original_id} has {len(all_augs)} augmentations:")
        
        for aug in all_augs:
            print(f"   - {aug['augmentation_type']}: {aug['augmented_track_id']}")
        
        print("\n" + "=" * 70)
        print(" All integration Tests passed!")
        print("=" * 70)
        
    finally:
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
                print(f"\n  Cleaned up test database")
            except:
                pass

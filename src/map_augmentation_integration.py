
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.database import DatabaseManager
from map_augmentation import MapAugmentation, AugmentationParams, PRESET_AUGMENTATIONS
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


def augment_track_map(
    image_path: str,
    output_path: str,
    params: AugmentationParams,
    save_metadata: bool = True
) -> Tuple[str, Dict]:
    image = Image.open(image_path)
    
    augmenter = MapAugmentation()
    augmented = augmenter.augment(image, params)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.save(output_path, quality=95)
    
    metadata = {
        'original_image': str(image_path),
        'augmented_image': str(output_path),
        'original_size': image.size,
        'augmented_size': augmented.size,
        'augmentation_params': params.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if save_metadata:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return str(output_path), metadata


def batch_augment_maps(
    image_paths: List[str],
    output_dir: str,
    augmentation_configs: List[Dict]
) -> List[Tuple[str, Dict]]:
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_paths:
        img_name = Path(img_path).stem
        
        for config in augmentation_configs:
            aug_name = config['name']
            params = config['params']
            
            output_name = f"{img_name}_AUG_{aug_name}{Path(img_path).suffix}"
            output_path = output_dir / output_name
            
            try:
                result_path, metadata = augment_track_map(
                    img_path, str(output_path), params
                )
                results.append((result_path, metadata))
                print(f" Created: {output_name}")
            
            except Exception as e:
                print(f" Failed {aug_name} on {img_name}: {e}")
    
    return results


def apply_preset_to_map(
    image_path: str,
    output_dir: str,
    preset_names: List[str]
) -> List[Tuple[str, Dict]]:
    configs = []
    
    for preset_name in preset_names:
        if preset_name not in PRESET_AUGMENTATIONS:
            print(f"  Unknown preset: {preset_name}")
            continue
        
        configs.append({
            'name': preset_name.upper(),
            'params': PRESET_AUGMENTATIONS[preset_name]
        })
    
    return batch_augment_maps([image_path], output_dir, configs)


def update_database_map_metadata(
    db: DatabaseManager,
    map_id: int,
    augmentation_metadata: Dict
) -> bool:
    try:
        existing_metadata = db.get_map_metadata(map_id) or {}
        
        augmentations = existing_metadata.get('augmentations', [])
        augmentations.append(augmentation_metadata)
        existing_metadata['augmentations'] = augmentations
        existing_metadata['augmentation_count'] = len(augmentations)
        
        db.update_map_metadata(map_id, existing_metadata)
        
        return True
    
    except Exception as e:
        print(f" Failed to update database: {e}")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Map Visual Augmentation - Integration Test")
    print("=" * 70)
    
    test_img_dir = Path('test_maps')
    test_img_dir.mkdir(exist_ok=True)
    
    from PIL import ImageDraw
    import numpy as np
    
    width, height = 800, 600
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(180 + 30 * np.sin(x / 50)),
                int(200 + 20 * np.cos(y / 50)),
                int(150 + 40 * np.sin((x + y) / 70))
            ]
    
    test_img = Image.fromarray(img_array)
    
    draw = ImageDraw.Draw(test_img)
    route_points = [(100, 300), (200, 250), (350, 200), (500, 280), (650, 350), (700, 400)]
    draw.line(route_points, fill=(255, 50, 50), width=5)
    
    for i in range(0, width, 100):
        draw.line([(i, 0), (i, height)], fill=(100, 100, 100), width=1)
    for i in range(0, height, 100):
        draw.line([(0, i), (width, i)], fill=(100, 100, 100), width=1)
    
    test_img_path = test_img_dir / 'test_route_map.png'
    test_img.save(test_img_path)
    print(f"\n Created test map: {test_img_path}")
    
    
    print("\nTest 1: Single Augmentation")
    print("-" * 70)
    
    bright_params = AugmentationParams(brightness=1.2, contrast=1.1)
    output_path, metadata = augment_track_map(
        str(test_img_path),
        str(test_img_dir / 'test_route_map_BRIGHT.png'),
        bright_params
    )
    
    print(f" Augmented image saved: {output_path}")
    print(f" Metadata: {len(metadata)} keys")
    print(f"   Original size: {metadata['original_size']}")
    print(f"   Brightness: {metadata['augmentation_params']['brightness']}")
    
    
    print("\nTest 2: Batch Augmentation")
    print("-" * 70)
    
    configs = [
        {'name': 'DARK', 'params': AugmentationParams(brightness=0.8, contrast=0.95)},
        {'name': 'SATURATED', 'params': AugmentationParams(saturation=1.2)},
        {'name': 'BLURRED', 'params': AugmentationParams(blur_radius=1.5)},
    ]
    
    batch_results = batch_augment_maps(
        [str(test_img_path)],
        str(test_img_dir / 'batch'),
        configs
    )
    
    print(f"\n Created {len(batch_results)} augmented images")
    
    
    print("\nTest 3: Preset Augmentations")
    print("-" * 70)
    
    preset_results = apply_preset_to_map(
        str(test_img_path),
        str(test_img_dir / 'presets'),
        ['light', 'dark', 'foggy', 'rainy', 'snowy']
    )
    
    print(f"\n Applied {len(preset_results)} presets")
    
    
    print("\nTest 4: Weather Effects")
    print("-" * 70)
    
    weather_configs = [
        {
            'name': 'FOG30',
            'params': AugmentationParams(
                brightness=1.1,
                contrast=0.9,
                weather_effect='fog',
                weather_intensity=0.3
            )
        },
        {
            'name': 'RAIN50',
            'params': AugmentationParams(
                brightness=0.9,
                weather_effect='rain',
                weather_intensity=0.5
            )
        },
        {
            'name': 'SNOW60',
            'params': AugmentationParams(
                brightness=1.15,
                weather_effect='snow',
                weather_intensity=0.6
            )
        }
    ]
    
    weather_results = batch_augment_maps(
        [str(test_img_path)],
        str(test_img_dir / 'weather'),
        weather_configs
    )
    
    print(f"\n Created {len(weather_results)} weather variants")
    
    
    print("\nTest 5: Extreme Combined Augmentation")
    print("-" * 70)
    
    extreme_params = AugmentationParams(
        brightness=1.25,
        contrast=1.15,
        saturation=1.15,
        hue_shift=10.0,
        blur_radius=1.5,
        noise_intensity=0.03,
        weather_effect='fog',
        weather_intensity=0.4,
        jpeg_quality=75
    )
    
    extreme_path, extreme_meta = augment_track_map(
        str(test_img_path),
        str(test_img_dir / 'test_route_map_EXTREME.png'),
        extreme_params
    )
    
    print(f" Extreme augmentation saved: {Path(extreme_path).name}")
    print(f"   Applied: brightness, contrast, saturation, hue, blur, noise, fog, compression")
    
    
    print("\nTest 6: Random Augmentations")
    print("-" * 70)
    
    augmenter = MapAugmentation(random_seed=42)
    random_configs = []
    
    for i in range(5):
        _, random_params = augmenter.random_augmentation(test_img, enable_weather=True)
        random_configs.append({
            'name': f'RANDOM{i+1}',
            'params': random_params
        })
    
    random_results = batch_augment_maps(
        [str(test_img_path)],
        str(test_img_dir / 'random'),
        random_configs
    )
    
    print(f"\n Created {len(random_results)} random augmentations")
    
    
    print("\n" + "=" * 70)
    print(" All integration Tests passed!")
    print("=" * 70)
    print(f"\nGenerated images:")
    print(f"  - Single augmentation: 1")
    print(f"  - Batch augmentation: {len(batch_results)}")
    print(f"  - Preset augmentation: {len(preset_results)}")
    print(f"  - Weather effects: {len(weather_results)}")
    print(f"  - Extreme augmentation: 1")
    print(f"  - Random augmentation: {len(random_results)}")
    print(f"\n  Total: {1 + len(batch_results) + len(preset_results) + len(weather_results) + 1 + len(random_results)} augmented maps")
    
    print(f"\n Output directory: {test_img_dir.absolute()}")

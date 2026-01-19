
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import json
import random
from datetime import datetime


MIN_BRIGHTNESS = 0.7
MAX_BRIGHTNESS = 1.3
MIN_CONTRAST = 0.8
MAX_CONTRAST = 1.2
MIN_SATURATION = 0.8
MAX_SATURATION = 1.2
MIN_HUE_SHIFT = -15
MAX_HUE_SHIFT = 15
MAX_BLUR_RADIUS = 2.0
MAX_NOISE_INTENSITY = 0.05
MIN_JPEG_QUALITY = 70
MAX_JPEG_QUALITY = 95


@dataclass
class AugmentationParams:
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    hue_shift: float = 0.0
    blur_radius: float = 0.0
    noise_intensity: float = 0.0
    jpeg_quality: Optional[int] = None
    weather_effect: Optional[str] = None
    weather_intensity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'hue_shift': self.hue_shift,
            'blur_radius': self.blur_radius,
            'noise_intensity': self.noise_intensity,
            'jpeg_quality': self.jpeg_quality,
            'weather_effect': self.weather_effect,
            'weather_intensity': self.weather_intensity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AugmentationParams':
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class MapAugmentation:
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.augmentation_history: List[AugmentationParams] = []
    
    def adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        if not (MIN_BRIGHTNESS <= factor <= MAX_BRIGHTNESS):
            raise ValueError(f"Brightness factor {factor} outside range [{MIN_BRIGHTNESS}, {MAX_BRIGHTNESS}]")
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        if not (MIN_CONTRAST <= factor <= MAX_CONTRAST):
            raise ValueError(f"Contrast factor {factor} outside range [{MIN_CONTRAST}, {MAX_CONTRAST}]")
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, image: Image.Image, factor: float) -> Image.Image:
        if not (MIN_SATURATION <= factor <= MAX_SATURATION):
            raise ValueError(f"Saturation factor {factor} outside range [{MIN_SATURATION}, {MAX_SATURATION}]")
        
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def adjust_hue(self, image: Image.Image, shift_degrees: float) -> Image.Image:
        if not (MIN_HUE_SHIFT <= shift_degrees <= MAX_HUE_SHIFT):
            raise ValueError(f"Hue shift {shift_degrees}° outside range [{MIN_HUE_SHIFT}, {MAX_HUE_SHIFT}]")
        
        if abs(shift_degrees) < 0.1:
            return image
        
        img_array = np.array(image.convert('RGB'))
        img_hsv = np.array(image.convert('HSV'))
        
        h_channel = img_hsv[:, :, 0].astype(np.int16)
        h_shift = int((shift_degrees / 360.0) * 256)
        h_channel = (h_channel + h_shift) % 256
        img_hsv[:, :, 0] = h_channel.astype(np.uint8)
        
        img_shifted = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
        return img_shifted
    
    def apply_blur(self, image: Image.Image, radius: float) -> Image.Image:
        if not (0.0 <= radius <= MAX_BLUR_RADIUS):
            raise ValueError(f"Blur radius {radius} outside range [0.0, {MAX_BLUR_RADIUS}]")
        
        if radius < 0.1:
            return image
        
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def add_gaussian_noise(self, image: Image.Image, intensity: float) -> Image.Image:
        if not (0.0 <= intensity <= MAX_NOISE_INTENSITY):
            raise ValueError(f"Noise intensity {intensity} outside range [0.0, {MAX_NOISE_INTENSITY}]")
        
        if intensity < 0.001:
            return image
        
        img_array = np.array(image).astype(np.float32)
        
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def apply_jpeg_compression(self, image: Image.Image, quality: int) -> Image.Image:
        if not (MIN_JPEG_QUALITY <= quality <= MAX_JPEG_QUALITY):
            raise ValueError(f"JPEG quality {quality} outside range [{MIN_JPEG_QUALITY}, {MAX_JPEG_QUALITY}]")
        
        import io
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer)
    
    def apply_fog_overlay(self, image: Image.Image, intensity: float = 0.3) -> Image.Image:
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"Fog intensity {intensity} outside range [0.0, 1.0]")
        
        if intensity < 0.05:
            return image
        
        fog = Image.new('RGBA', image.size, (255, 255, 255, int(intensity * 200)))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        result = Image.alpha_composite(image, fog)
        
        return result.convert('RGB')
    
    def apply_rain_overlay(self, image: Image.Image, intensity: float = 0.5) -> Image.Image:
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"Rain intensity {intensity} outside range [0.0, 1.0]")
        
        if intensity < 0.05:
            return image
        
        darkened = self.adjust_brightness(image, 1.0 - intensity * 0.2)
        
        rain_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(rain_layer)
        
        num_streaks = int(intensity * 500)
        
        for _ in range(num_streaks):
            x = random.randint(0, image.width)
            y = random.randint(0, image.height)
            length = random.randint(10, 30)
            alpha = random.randint(50, 150)
            
            x_offset = random.randint(-2, 2)
            draw.line(
                [(x, y), (x + x_offset, y + length)],
                fill=(200, 200, 255, alpha),
                width=1
            )
        
        if darkened.mode != 'RGBA':
            darkened = darkened.convert('RGBA')
        
        result = Image.alpha_composite(darkened, rain_layer)
        
        return result.convert('RGB')
    
    def apply_snow_overlay(self, image: Image.Image, intensity: float = 0.5) -> Image.Image:
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"Snow intensity {intensity} outside range [0.0, 1.0]")
        
        if intensity < 0.05:
            return image
        
        brightened = self.adjust_brightness(image, 1.0 + intensity * 0.15)
        
        snow_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(snow_layer)
        
        num_flakes = int(intensity * 800)
        
        for _ in range(num_flakes):
            x = random.randint(0, image.width)
            y = random.randint(0, image.height)
            radius = random.randint(1, 3)
            alpha = random.randint(150, 255)
            
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, fill=(255, 255, 255, alpha))
        
        if brightened.mode != 'RGBA':
            brightened = brightened.convert('RGBA')
        
        result = Image.alpha_composite(brightened, snow_layer)
        
        return result.convert('RGB')
    
    def apply_weather_effect(self, image: Image.Image, weather_type: str, 
                            intensity: float = 0.5) -> Image.Image:
        weather_type = weather_type.lower()
        
        if weather_type == 'fog':
            return self.apply_fog_overlay(image, intensity)
        elif weather_type == 'rain':
            return self.apply_rain_overlay(image, intensity)
        elif weather_type == 'snow':
            return self.apply_snow_overlay(image, intensity)
        else:
            raise ValueError(f"Unknown weather type: {weather_type}")
    
    def augment(self, image: Image.Image, params: AugmentationParams) -> Image.Image:
        result = image.copy()
        
        if abs(params.brightness - 1.0) > 0.01:
            result = self.adjust_brightness(result, params.brightness)
        
        if abs(params.contrast - 1.0) > 0.01:
            result = self.adjust_contrast(result, params.contrast)
        
        if abs(params.saturation - 1.0) > 0.01:
            result = self.adjust_saturation(result, params.saturation)
        
        if abs(params.hue_shift) > 0.1:
            result = self.adjust_hue(result, params.hue_shift)
        
        if params.blur_radius > 0.1:
            result = self.apply_blur(result, params.blur_radius)
        
        if params.noise_intensity > 0.001:
            result = self.add_gaussian_noise(result, params.noise_intensity)
        
        if params.weather_effect and params.weather_intensity > 0.05:
            result = self.apply_weather_effect(result, params.weather_effect, params.weather_intensity)
        
        if params.jpeg_quality is not None:
            result = self.apply_jpeg_compression(result, params.jpeg_quality)
        
        self.augmentation_history.append(params)
        
        return result
    
    def random_augmentation(self, image: Image.Image, 
                           enable_weather: bool = True) -> Tuple[Image.Image, AugmentationParams]:
        params = AugmentationParams(
            brightness=random.uniform(MIN_BRIGHTNESS, MAX_BRIGHTNESS),
            contrast=random.uniform(MIN_CONTRAST, MAX_CONTRAST),
            saturation=random.uniform(MIN_SATURATION, MAX_SATURATION),
            hue_shift=random.uniform(MIN_HUE_SHIFT, MAX_HUE_SHIFT),
            blur_radius=random.uniform(0.0, MAX_BLUR_RADIUS),
            noise_intensity=random.uniform(0.0, MAX_NOISE_INTENSITY),
            jpeg_quality=random.randint(MIN_JPEG_QUALITY, MAX_JPEG_QUALITY)
        )
        
        if enable_weather and random.random() < 0.3:
            weather_types = ['fog', 'rain', 'snow']
            params.weather_effect = random.choice(weather_types)
            params.weather_intensity = random.uniform(0.2, 0.6)
        
        augmented = self.augment(image, params)
        
        return augmented, params
    
    def clear_history(self):
        self.augmentation_history.clear()


PRESET_AUGMENTATIONS = {
    'light': AugmentationParams(
        brightness=1.1,
        contrast=1.05,
        saturation=1.05
    ),
    'dark': AugmentationParams(
        brightness=0.85,
        contrast=0.95,
        saturation=0.95
    ),
    'high_contrast': AugmentationParams(
        brightness=1.0,
        contrast=1.2,
        saturation=1.1
    ),
    'low_saturation': AugmentationParams(
        brightness=1.0,
        contrast=1.0,
        saturation=0.8
    ),
    'warm_tone': AugmentationParams(
        brightness=1.05,
        saturation=1.1,
        hue_shift=5.0
    ),
    'cool_tone': AugmentationParams(
        brightness=0.95,
        saturation=1.05,
        hue_shift=-5.0
    ),
    'foggy': AugmentationParams(
        brightness=1.1,
        contrast=0.9,
        weather_effect='fog',
        weather_intensity=0.4
    ),
    'rainy': AugmentationParams(
        brightness=0.9,
        contrast=0.95,
        weather_effect='rain',
        weather_intensity=0.5
    ),
    'snowy': AugmentationParams(
        brightness=1.15,
        contrast=0.95,
        weather_effect='snow',
        weather_intensity=0.5
    ),
    'compressed': AugmentationParams(
        jpeg_quality=75
    ),
    'slightly_blurred': AugmentationParams(
        blur_radius=1.0
    )
}


if __name__ == '__main__':
    print("=" * 70)
    print("Map Visual Augmentation Module - Test")
    print("=" * 70)
    
    test_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    for i in range(0, 640, 40):
        draw.line([(i, 0), (i, 480)], fill=(80, 130, 180), width=2)
    for i in range(0, 480, 40):
        draw.line([(0, i), (640, i)], fill=(80, 130, 180), width=2)
    
    augmenter = MapAugmentation(random_seed=42)
    
    print("\nTest 1: Brightness Adjustment")
    print("-" * 70)
    bright = augmenter.adjust_brightness(test_image, 1.2)
    print(f" Brightness increased: {test_image.size} → {bright.size}")
    
    print("\nTest 2: Contrast Adjustment")
    print("-" * 70)
    contrast = augmenter.adjust_contrast(test_image, 1.15)
    print(f" Contrast enhanced: {test_image.size} → {contrast.size}")
    
    print("\nTest 3: Saturation Adjustment")
    print("-" * 70)
    saturated = augmenter.adjust_saturation(test_image, 1.2)
    print(f" Saturation increased: {test_image.size} → {saturated.size}")
    
    print("\nTest 4: Hue Shift")
    print("-" * 70)
    hue_shifted = augmenter.adjust_hue(test_image, 10.0)
    print(f" Hue shifted +10°: {test_image.size} → {hue_shifted.size}")
    
    print("\nTest 5: Gaussian Blur")
    print("-" * 70)
    blurred = augmenter.apply_blur(test_image, 1.5)
    print(f" Blur applied (radius=1.5): {test_image.size} → {blurred.size}")
    
    print("\nTest 6: Gaussian Noise")
    print("-" * 70)
    noisy = augmenter.add_gaussian_noise(test_image, 0.02)
    print(f" Noise added (2%): {test_image.size} → {noisy.size}")
    
    print("\nTest 7: Weather Effects")
    print("-" * 70)
    foggy = augmenter.apply_fog_overlay(test_image, 0.3)
    print(f" Fog overlay: {test_image.size} → {foggy.size}")
    
    rainy = augmenter.apply_rain_overlay(test_image, 0.5)
    print(f" Rain overlay: {test_image.size} → {rainy.size}")
    
    snowy = augmenter.apply_snow_overlay(test_image, 0.5)
    print(f" Snow overlay: {test_image.size} → {snowy.size}")
    
    print("\nTest 8: Full Augmentation Pipeline")
    print("-" * 70)
    params = AugmentationParams(
        brightness=1.1,
        contrast=1.05,
        saturation=1.1,
        hue_shift=5.0,
        blur_radius=0.5,
        noise_intensity=0.01,
        jpeg_quality=85
    )
    augmented = augmenter.augment(test_image, params)
    print(f" Full pipeline applied: {test_image.size} → {augmented.size}")
    
    print("\nTest 9: Random Augmentation")
    print("-" * 70)
    random_aug, random_params = augmenter.random_augmentation(test_image)
    print(f" Random augmentation: brightness={random_params.brightness:.2f}, " +
          f"contrast={random_params.contrast:.2f}")
    
    print("\nTest 10: Preset Augmentations")
    print("-" * 70)
    for preset_name, preset_params in list(PRESET_AUGMENTATIONS.items())[:3]:
        preset_result = augmenter.augment(test_image, preset_params)
        print(f" Preset '{preset_name}': {test_image.size} → {preset_result.size}")
    
    print("\n" + "=" * 70)
    print(f" All transformations completed: {len(augmenter.augmentation_history)} in history")
    print("=" * 70)

def _add_missing_methods():
    
    def rotate_image(self, image: Image.Image, angle: float, expand: bool = False) -> Image.Image:
        return image.rotate(angle, expand=expand, resample=Image.BICUBIC)
    
    def shift_image(self, image: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
        return image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            Image.BICUBIC
        )
    
    MapAugmentation.rotate_image = rotate_image
    MapAugmentation.shift_image = shift_image


_add_missing_methods()

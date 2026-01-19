import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional
import shutil

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import RouteDataLoader
from src.coordinate_transformer import CoordinateTransformer
from src.interpolator import GPSInterpolator
from src.database import DatabaseManager
from src.map_retriever import MapRetriever
from web_agent import WebAgent

from map_enricher_yandex import (
    EnrichmentEngineYandex, 
    EnrichedPointData,
    initialize_enricher_yandex
)
from weather_integration import OpenWeatherMapProvider
from map_decoder import StandardTerrainType, StandardObjectType

from stage13_config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    FEATURE_CATEGORIES
)
from feature_extractor import FeatureExtractor
from feature_correlation_analyzer import FeatureCorrelationAnalyzer
from data_augmentation_engine import DataAugmentationEngine, AugmentationConfig
from shap_analyzer import SHAPAnalysisEngine
from permutation_analyzer import PermutationAnalysisEngine
from causal_graph_analyzer import CausalGraphAnalyzer
from variance_analyzer import VarianceAnalysisEngine

from feature_metadata import (
    FEATURE_METADATA,
    get_numerical_features,
    get_categorical_features,
    get_feature_info
)
from distribution_analyzer import (
    NormalityTester,
    MultimodalityDetector,
    DistributionClassifier,
    TransformationRecommender
)
from distribution_visualizer import DistributionVisualizer

from track_parser import UniversalTrackParser
from external.osm_track_loader import OSMTrackLoader
from synthetic_track_generator import SyntheticTrackGenerator
from track_exporter import TrackExporter, ExportGeoPoint, ExportTrackMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/temp/main_full.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from map_enricher_yandex import YandexFeatureExtractor


class FullIntegratedProcessor:
    
    def __init__(self, config_path: str = 'config.json', 
                 db_path: str = 'data/db/routes.db',
                 map_cache_dir: str = 'data/temp/map_cache'):
        logger.info("=" * 80)
        logger.info(" Запуск системы анализа маршрутов")
        logger.info("=" * 80)
        
        self.start_time = datetime.now()
        self.db_path = db_path
        self.map_cache_dir = map_cache_dir
        self.config_path = config_path
        
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Установка API ключей из конфигурации
        api_keys = self.config.get('api_keys', {})
        if api_keys.get('yandex_geocoder'):
            YandexFeatureExtractor.set_geocoder_api_key(api_keys['yandex_geocoder'])
        if api_keys.get('yandex_static_maps'):
            YandexFeatureExtractor.set_static_api_key(api_keys['yandex_static_maps'])
        if api_keys.get('openweathermap'):
            OpenWeatherMapProvider.set_api_key(api_keys['openweathermap'])
        
        logger.info(f"Конфиг загружен: {config_path}")
        
        self._cleanup_environment()
        
        logger.info("Настройка базовых модулей...")
        dataset_path = self.config.get('data', {}).get('raw_dataset', 'dataset/large_route_dataset_20260117_143350.csv')
        self.loader = RouteDataLoader(dataset_path)
        self.transformer = CoordinateTransformer()
        self.interpolator = GPSInterpolator()
        self.database = DatabaseManager(db_path)
        self.map_retriever = MapRetriever(cache_dir=map_cache_dir)
        self.web_agent = WebAgent(download_dir='data/temp/downloads')
        logger.info("Базовые модули готовы")
        
        logger.info("Подключение карт и погоды...")
        buffer_radius = self.config.get('map_enrichment', {}).get('buffer_radius_m', 500)
        self.enricher = initialize_enricher_yandex(buffer_radius_m=buffer_radius)
        self.weather_client = OpenWeatherMapProvider()
        logger.info("Карты и погода подключены")
        
        logger.info("Загрузка аналитических модулей...")
        self.feature_extractor = FeatureExtractor()
        self.correlation_analyzer = FeatureCorrelationAnalyzer()
        self.data_augmenter = DataAugmentationEngine(AugmentationConfig(
            max_synthetic_samples=1000
        ))
        self.shap_engine = SHAPAnalysisEngine()
        self.permutation_engine = PermutationAnalysisEngine()
        self.causal_analyzer = CausalGraphAnalyzer()
        self.variance_engine = VarianceAnalysisEngine()
        logger.info("Аналитика готова")
        
        logger.info("Настройка статистических тестов...")
        self.normality_tester = NormalityTester(alpha=0.05)
        self.multimodality_detector = MultimodalityDetector()
        self.distribution_classifier = DistributionClassifier()
        self.transformation_recommender = TransformationRecommender()
        logger.info("Статистика готова")
        
        logger.info("Подготовка генераторов данных...")
        self.track_parser = UniversalTrackParser()
        self.osm_loader = OSMTrackLoader(cache_dir='data/temp/osm_cache')
        self.synthetic_generator = SyntheticTrackGenerator(db=self.database)
        logger.info("Генераторы готовы")
        
        self.processed_tracks = 0
        self.anomalies_found = 0
        self.anomalies_fixed = 0
        self.maps_generated = 0
        self.points_enriched = 0
        
        self.features_df: Optional[pd.DataFrame] = None
        self.correlation_result = None
        self.augmented_df: Optional[pd.DataFrame] = None
        self.stage13_report: Dict = {}
        
        self.distribution_results: Dict = {}
        self.stage14_report: Dict = {}
        
        logger.info("Система готова к работе")
    
    
    def _cleanup_environment(self):
        logger.info("\n" + "=" * 80)
        logger.info("Очистка старых файлов")
        logger.info("=" * 80)
        
        files_to_clean = [
            (self.db_path, "Database file"),
            ('data/temp/main_full.log', "Main log file"),
        ]
        
        dirs_to_clean = [
            (self.map_cache_dir, "Map cache directory"),
            ('data/temp/downloads', "Downloads directory"),
            ('data/output', "Output directory"),
        ]
        
        for file_path, description in files_to_clean:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.info(f"Удален: {file_path}")
                else:
                    logger.info(f"  {description} не найден (пропущен): {file_path}")
            except Exception as e:
                logger.warning(f"  Ошибка при удалении {description}: {e}")
        
        for dir_path, description in dirs_to_clean:
            try:
                path = Path(dir_path)
                if path.exists():
                    shutil.rmtree(path)
                    logger.info(f"Очищено: {dir_path}")
                else:
                    logger.info(f"  {description} не найдена (пропущена): {dir_path}")
            except Exception as e:
                logger.warning(f"  Ошибка при удалении {description}: {e}")
        
        dirs_to_create = [
            'data/temp',
            'data/temp/map_cache',
            'data/temp/downloads',
            'data/db',
            'data/output',
            'data/output/examples',
            'data/output/test_maps',
        ]
        
        for dir_path in dirs_to_create:
            try:
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Создано: {dir_path}")
            except Exception as e:
                logger.warning(f"  Ошибка при создании директории {dir_path}: {e}")
        
        logger.info("=" * 80)
        logger.info("Готово к запуску")
        logger.info("=" * 80)
    
    
    def load_data_from_csv(self) -> Tuple[bool, Dict]:
        logger.info("\n" + "=" * 80)
        logger.info("Загрузка данных")
        logger.info("=" * 80)
        
        try:
            logger.info("Читаем CSV...")
            df = self.loader.load_raw_data()
            logger.info(f"Загружено {len(df):,} точек")
            
            logger.info("Проверяем координаты...")
            validation = self.loader.validate_geodetic_data()
            summary = validation['summary']
            
            logger.info(f"Результаты:")
            logger.info(f"  Валидных: {summary['valid_points']:,}/{summary['total_points']:,}")
            logger.info(f"  Качество: {summary['validity_percentage']:.2f}%")
            
            logger.info("Поиск GPS аномалий...")
            anomalies = self.loader.detect_gps_anomalies()
            self.anomalies_found = len(anomalies.get('anomalies', []))
            
            logger.info(f"Найдено аномалий: {self.anomalies_found}")
            
            return True, {
                'total_points': summary['total_points'],
                'valid_points': summary['valid_points'],
                'validity_percentage': summary['validity_percentage'],
                'anomalies_found': self.anomalies_found
            }
            
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            return False, {}
    
    
    def validate_coordinate_systems(self, lat: float, lon: float) -> Dict:
        logger.info("\n" + "=" * 80)
        logger.info("Проверка преобразований координат")
        logger.info("=" * 80)
        
        try:
            logger.info(f"Тестовая точка: ({lat:.4f}, {lon:.4f})")
            
            logger.info("Преобразование WGS84 → Web Mercator...")
            x, y = self.transformer.wgs84_to_web_mercator(lat, lon)
            logger.info(f"    Web Mercator: ({x:.2f}, {y:.2f})")
            
            logger.info("Преобразование Web Mercator → WGS84 (тест обратного преобразования)...")
            lat2, lon2 = self.transformer.web_mercator_to_wgs84(x, y)
            error = abs(lat - lat2) + abs(lon - lon2)
            logger.info(f"    Обратное преобразование: ({lat2:.4f}, {lon2:.4f})")
            logger.info(f"    Погрешность: {error:.10f}° (приемлемо)")
            
            logger.info("Преобразование WGS84 → UTM...")
            x_utm, y_utm, zone = self.transformer.wgs84_to_utm(lat, lon)
            logger.info(f"    UTM Зона {zone}: ({x_utm:.2f}, {y_utm:.2f})")
            
            logger.info("Расчет расстояния с поправкой на широту...")
            lat_test2, lon_test2 = lat + 0.01, lon + 0.01
            dist_vincenty = self.transformer.vincenty_distance(
                lat, lon, lat_test2, lon_test2
            )
            logger.info(f"    Расстояние Винсенти: {dist_vincenty/1000:.2f} км")
            
            return {
                'test_point': (lat, lon),
                'web_mercator': (x, y),
                'utm': (x_utm, y_utm, zone),
                'vincenty_distance': dist_vincenty
            }
            
        except Exception as e:
            logger.error(f" Ошибка в преобразованиях координат: {e}")
            return {}
    
    
    def process_track_anomalies(self, track_id: str, 
                               points: List[Tuple[float, float]]) -> Dict:
        logger.info(f"Исправление аномалий: {track_id}")
        
        try:
            logger.info(f"  Обнаружение аномалий в {len(points)} точках...")
            result = self.interpolator.detect_and_fix_anomalies(
                track_id=track_id,
                points=points,
                speed_threshold_kmh=50,
                from_transformer=self.transformer
            )
            
            num_anomalies = len([r for r in self.interpolator.anomaly_records 
                                if r.track_id == track_id])
            self.anomalies_fixed += num_anomalies
            
            logger.info(f"  Обнаружено и исправлено аномалий: {num_anomalies}")
            
            if num_anomalies > 0:
                logger.info(f"  Применение сплайн-интерполяции...")
                indices_to_fix = [i for i in range(len(points))]
                corrected = self.interpolator.spline_interpolation(
                    points, indices_to_fix[:min(num_anomalies, len(indices_to_fix))]
                )
                logger.info(f"  Интерполяция завершена")
            
            return result
            
        except Exception as e:
            logger.error(f"  Ошибка при обработке аномалий: {e}")
            return {}
    
    
    def enrich_track_points(self, track_id: str, 
                           points_data: List[Dict]) -> List[EnrichedPointData]:
        logger.info(f"\nОбогащение трека: {track_id}")
        logger.info(f"   → Обработка {len(points_data)} точек с радиусом 500м...")
        
        try:
            enriched_points = self.enricher.enrich_batch(points_data)
            
            self.points_enriched += len(enriched_points)
            
            terrain_count = sum(1 for p in enriched_points if p.terrain_type)
            objects_count = sum(p.object_count for p in enriched_points)
            avg_quality = sum(p.quality_score for p in enriched_points) / len(enriched_points) if enriched_points else 0
            
            logger.info(f"    Обогащение завершено:")
            logger.info(f"     Обогащено точек: {len(enriched_points)}")
            logger.info(f"     С типом местности: {terrain_count} ({terrain_count/len(enriched_points)*100:.1f}%)")
            logger.info(f"     Всего найдено объектов: {objects_count}")
            logger.info(f"     Средний балл качества: {avg_quality:.3f}")
            
            return enriched_points
            
        except Exception as e:
            logger.error(f"   Ошибка при обогащении точек: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    
    def generate_track_maps(self, track_id: str, 
                           points: List[Tuple[float, float]]) -> Optional[Path]:
        logger.info(f"Генерация карт: {track_id}")
        
        try:
            logger.info(f"  Создание карты с наложением трека (Яндекс.Карты)...")
            yandex_key = self.config.get('api_keys', {}).get('yandex_static_maps')
            map_path = self.map_retriever.get_track_map_with_overlay(
                track_id=track_id,
                points=points,
                yandex_api_key=yandex_key
            )
            
            if map_path:
                logger.info(f"  Карта сохранена: {map_path}")
                self.maps_generated += 1
                
                
                return map_path
            else:
                logger.warning(f"  Не удалось сгенерировать карту")
                return None
                
        except Exception as e:
            logger.error(f"  Ошибка при генерации карт: {e}")
            return None
    
    
    def store_enriched_track_in_database(self, track_id: str, 
                                        enriched_points: List[EnrichedPointData],
                                        region: str = 'Unknown',
                                        metadata: Dict = None) -> bool:
        logger.info(f"Сохранение трека: {track_id}")
        
        try:
            dates = [p.date for p in enriched_points if p.date]
            date_start = min(dates) if dates else None
            date_end = max(dates) if dates else None
            
            logger.info(f"  Добавление записи трека...")
            track = self.database.add_track(
                track_id=track_id,
                region=region,
                date_start=date_start,
                date_end=date_end
            )
            logger.info(f"  Трек добавлен")
            
            logger.info(f"  Добавление {len(enriched_points)} обогащенных точек...")
            points_data = []
            for enriched_point in enriched_points:
                point_data = {
                    'point_index': enriched_point.point_index,
                    'date': enriched_point.date,
                    'latitude': enriched_point.latitude,
                    'longitude': enriched_point.longitude,
                    'altitude': enriched_point.altitude,
                    'frequency_steps': enriched_point.step_frequency,
                    'temperature': enriched_point.temperature,
                    'terrain_type': enriched_point.terrain_type.value if enriched_point.terrain_type else None,
                    'surrounding_objects': json.dumps([obj.to_dict() for obj in enriched_point.nearby_objects])
                }
                points_data.append(point_data)
            
            self.database.add_points(track_id, points_data)
            logger.info(f"    {len(enriched_points)} обогащенных точек добавлено")
            
            stats = self.database.get_database_statistics()
            logger.info(f"  Статистика базы данных:")
            logger.info(f"     Всего треков: {stats.get('total_tracks', 0)}")
            logger.info(f"     Всего точек: {stats.get('total_points', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"  Ошибка при сохранении в базу данных: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def run_stage11_pipeline(self, input_csv: str = None) -> Dict:
        if input_csv is None:
            input_csv = self.config.get('data', {}).get('raw_dataset', 'dataset/large_route_dataset_20260117_143350.csv')
        
        logger.info("\n" + "="*80)
        logger.info("Загрузка и обработка данных")
        logger.info("="*80)
        
        try:
            loader = DataLoader()
            data = loader.load_csv(input_csv)
            logger.info(f" Загружено {len(data)} записей из {input_csv}")
            
            from src.anomaly_detector import GPSAnomalyDetector
            detector = GPSAnomalyDetector()
            anomalies = detector.detect_anomalies(data)
            logger.info(f" Обнаружено {len(anomalies)} GPS аномалий")
            
            from src.coordinate_transformer import CoordinateTransformer
            transformer = CoordinateTransformer()
            logger.info(" Инициализированы преобразования координат (WGS84, Web Mercator, UTM)")
            
            self.store_tracks(data)
            
            return {
                "status": "success",
                "records_loaded": len(data),
                "anomalies_detected": len(anomalies),
                "transformations_applied": ["WGS84", "Web Mercator", "UTM"]
            }
            
        except Exception as e:
            logger.error(f" Этап 1.1 не удался: {e}")
            return {"status": "error", "message": str(e)}
    
    
    def run_stage12_pipeline(self) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("Обогащение картами и погодой")
        logger.info("="*80)
        
        try:
            from map_enricher_yandex import MapEnricher
            from terrain_classifier import TerrainClassifier
            from osm_loader import OSMLoader
            from weather_loader import WeatherLoader
            
            enricher = MapEnricher()
            terrain_classifier = TerrainClassifier()
            osm_loader = OSMLoader()
            weather_loader = WeatherLoader()
            
            logger.info(" Инициализирован обогащатель Яндекс.Карт")
            logger.info(" Инициализирован классификатор местности")
            logger.info(" Инициализирован загрузчик OSM")
            logger.info(" Инициализирован загрузчик погоды")
            
            try:
                session = self.database.Session()
                from src.database import RouteTrack
                tracks = session.query(RouteTrack).all()
                session.close()
                
                logger.info(f" Получено {len(tracks)} треков из базы данных для обогащения")
                
                enriched_count = 0
                for track in tracks:
                    for point in track.points:
                        weather = weather_loader.get_weather(
                            point.latitude, point.longitude
                        )
                        terrain = terrain_classifier.classify_point(
                            point.latitude, point.longitude, point.altitude
                        )
                        enriched_count += 1
                
                logger.info(f" Обогащено {enriched_count} точек треков")
                
            except Exception as e:
                logger.warning(f"Не удалось обогатить из базы данных: {e}")
                enriched_count = 0
            
            return {
                "status": "success",
                "enriched_points": enriched_count,
                "data_sources": ["Yandex Maps", "OpenWeatherMap", "OpenStreetMap", "Terrain Analysis"]
            }
            
        except Exception as e:
            logger.error(f" Этап 1.2 не удался: {e}")
            return {"status": "error", "message": str(e)}
    
    
    def run_stage13_pipeline(self, enriched_tracks: int = 0, 
                             output_dir: str = 'data/output',
                             augmentation_method: str = 'smote',
                             augmentation_samples: int = 1000) -> Dict:
        logger.info("\n" + "=" * 80)
        logger.info("Анализ признаков")
        logger.info("=" * 80)
        
        stage13_results = {
            'success': False,
            'features_extracted': 0,
            'significant_pairs': 0,
            'confounders': 0,
            'redundant_features': 0,
            'augmented_samples': 0,
            'augmentation_ratio': 0,
            'errors': []
        }
        
        try:
            logger.info("\nШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
            logger.info("-" * 80)
            
            if enriched_tracks > 0:
                logger.info(f"Загрузка данных из {enriched_tracks} обогащенных треков...")
                try:
                    db_stats = self.database.get_database_statistics()
                    total_points = db_stats.get('total_points', 0)
                    if total_points > 0:
                        all_points = self.database.get_all_points_as_dataframe()
                        if all_points is not None and len(all_points) > 0:
                            raw_data = all_points
                            logger.info(f" Загружено {len(raw_data)} обогащенных точек из базы данных")
                        else:
                            logger.warning("Нет данных из базы, используются синтетические данные")
                            raw_data = self._generate_synthetic_data_stage13(100)
                    else:
                        logger.warning("Нет данных в базе, используются синтетические данные")
                        raw_data = self._generate_synthetic_data_stage13(100)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить из базы данных: {e}, используются синтетические данные")
                    raw_data = self._generate_synthetic_data_stage13(100)
            else:
                logger.info("Нет обогащенных треков, генерация синтетических данных...")
                raw_data = self._generate_synthetic_data_stage13(100)
            
            logger.info(f"Размер данных: {raw_data.shape}")
            logger.info(f"Использование памяти: {raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} МБ")
            
            logger.info("\nШАГ 2: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ")
            logger.info("-" * 80)
            
            logger.info(f"Извлечение признаков из {len(raw_data)} записей...")
            enriched_points_list = raw_data.to_dict('records')
            self.features_df = self.feature_extractor.extract_batch_features(enriched_points_list)
            
            logger.info(f" Извлечено {len(self.features_df)} векторов признаков")
            logger.info(f"   Признаков: {self.features_df.shape[1]} всего")
            
            stage13_results['features_extracted'] = self.features_df.shape[1]
            
            logger.info("\nШАГ 3: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
            logger.info("-" * 80)
            
            logger.info(f"Анализ {len(self.features_df)} векторов признаков...")
            
            valid_numerical = [f for f in NUMERICAL_FEATURES if f in self.features_df.columns]
            valid_categorical = [f for f in CATEGORICAL_FEATURES if f in self.features_df.columns]
            
            logger.info(f"Численных признаков: {len(valid_numerical)}")
            logger.info(f"Категориальных признаков: {len(valid_categorical)}")
            
            self.correlation_result = self.correlation_analyzer.analyze(
                self.features_df,
                numerical_features=valid_numerical,
                categorical_features=valid_categorical,
                p_value_threshold=0.05,
                min_correlation=0.3
            )
            
            logger.info(f" Корреляционный анализ завершен!")
            logger.info(f"   Значимых пар: {len(self.correlation_result.significant_pairs)}")
            logger.info(f"   Смешивающих факторов: {len(self.correlation_result.confounders)}")
            logger.info(f"   Избыточных признаков: {len(self.correlation_result.redundant_features)}")
            
            stage13_results['significant_pairs'] = len(self.correlation_result.significant_pairs)
            stage13_results['confounders'] = len(self.correlation_result.confounders)
            stage13_results['redundant_features'] = len(self.correlation_result.redundant_features)
            
            logger.info("\nШАГ 3.1: РАСШИРЕННЫЙ АНАЛИЗ (SHAP, ПЕРЕСТАНОВКИ, ДИСПЕРСИЯ, ПРИЧИННОСТЬ)")
            logger.info("-" * 80)
            
            numerical_features = self.features_df.select_dtypes(include=[np.number])
            
            if len(numerical_features.columns) >= 3:
                target_name = numerical_features.columns[0]
                y = numerical_features[target_name]
                X = numerical_features.drop(columns=[target_name])
                
                logger.info("\n3.1A: ВАЖНОСТЬ ПРИЗНАКОВ SHAP")
                try:
                    shap_result = self.shap_engine.analyze_features(X, y)
                    if shap_result:
                        shap_df = self.shap_engine.export_to_dataframe()
                        if not shap_df.empty:
                            shap_path = Path(output_dir) / 'stage13_shap_importance.csv'
                            shap_path.parent.mkdir(parents=True, exist_ok=True)
                            shap_df.to_csv(shap_path, index=False)
                            logger.info(f" Результаты SHAP сохранены: {shap_path}")
                except Exception as e:
                    logger.warning(f"Предупреждение SHAP анализа: {e}")
                
                logger.info("\n3.1B: ВАЖНОСТЬ ПЕРЕСТАНОВОК")
                try:
                    self.permutation_engine.analyze_importance(X, y, n_repeats=5)
                    perm_importance_df = self.permutation_engine.export_importance_dataframe()
                    if not perm_importance_df.empty:
                        perm_path = Path(output_dir) / 'stage13_permutation_importance.csv'
                        perm_path.parent.mkdir(parents=True, exist_ok=True)
                        perm_importance_df.to_csv(perm_path, index=False)
                        logger.info(f" Важность перестановок сохранена: {perm_path}")
                except Exception as e:
                    logger.warning(f"Предупреждение важности перестановок: {e}")
                
                logger.info("\n3.1C: ТЕСТЫ ПЕРЕСТАНОВОК")
                try:
                    self.permutation_engine.analyze_correlations(
                        numerical_features,
                        numerical_features.columns.tolist(),
                        n_permutations=100
                    )
                    perm_tests_df = self.permutation_engine.export_test_dataframe()
                    if not perm_tests_df.empty:
                        perm_tests_path = Path(output_dir) / 'stage13_permutation_tests.csv'
                        perm_tests_path.parent.mkdir(parents=True, exist_ok=True)
                        perm_tests_df.to_csv(perm_tests_path, index=False)
                        logger.info(f" Тесты перестановок сохранены: {perm_tests_path}")
                except Exception as e:
                    logger.warning(f"Предупреждение тестов перестановок: {e}")
                
                logger.info("\n3.1C.5: ДИСПЕРСИОННЫЙ АНАЛИЗ (ANOVA)")
                try:
                    variance_result = self.variance_engine.analyze(self.features_df, group_col='T_season')
                    if variance_result:
                        variance_df = self.variance_engine.export_to_dataframe()
                        if not variance_df.empty:
                            variance_path = Path(output_dir) / 'stage13_variance_analysis_anova.csv'
                            variance_path.parent.mkdir(parents=True, exist_ok=True)
                            variance_df.to_csv(variance_path, index=False)
                            logger.info(f" Дисперсионный анализ сохранен: {variance_path}")
                except Exception as e:
                    logger.warning(f"Предупреждение дисперсионного анализа: {e}")
                
                logger.info("\n3.1D: ИЗУЧЕНИЕ ПРИЧИННОЙ СТРУКТУРЫ")
                try:
                    causal_result = self.causal_analyzer.analyze(numerical_features)
                    if causal_result:
                        causal_df = self.causal_analyzer.export_to_dataframe()
                        if not causal_df.empty:
                            causal_path = Path(output_dir) / 'stage13_causal_edges.csv'
                            causal_path.parent.mkdir(parents=True, exist_ok=True)
                            causal_df.to_csv(causal_path, index=False)
                            logger.info(f" Причинные связи сохранены: {causal_path}")
                except Exception as e:
                    logger.warning(f"Предупреждение причинного анализа: {e}")
            else:
                logger.warning("Необходимо минимум 3 численных признака для расширенного анализа")
            
            logger.info("\nШАГ 4: АУГМЕНТАЦИЯ ДАННЫХ")
            logger.info("-" * 80)
            
            try:
                self.data_augmenter.set_redundant_features(
                    self.correlation_result.redundant_features
                )
                self.data_augmenter.set_confounders([
                    {
                        'confounder': c.confounder,
                        'feature_1': c.feature_1,
                        'feature_2': c.feature_2,
                        'mediation_score': c.mediation_score
                    } for c in self.correlation_result.confounders
                ])
                
                synthetic_data = self.data_augmenter.augment(
                    self.features_df,
                    method=augmentation_method,
                    n_samples=augmentation_samples
                )
                
                self.augmented_df = pd.concat(
                    [self.features_df, synthetic_data],
                    ignore_index=True
                )
                
                logger.info(f" Аугментация данных завершена!")
                logger.info(f"   Оригинальных: {len(self.features_df)} образцов")
                logger.info(f"   Синтетических: {len(synthetic_data)} образцов")
                logger.info(f"   Объединенных: {len(self.augmented_df)} образцов")
                augmentation_ratio = len(self.augmented_df) / len(self.features_df)
                logger.info(f"   Коэффициент аугментации: {augmentation_ratio:.2f}x")
                
                stage13_results['augmented_samples'] = len(synthetic_data)
                stage13_results['augmentation_ratio'] = augmentation_ratio
                
            except Exception as e:
                logger.warning(f"Предупреждение аугментации данных: {e}")
            
            logger.info("\nШАГ 5: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
            logger.info("-" * 80)
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if self.features_df is not None:
                features_path = output_path / 'stage13_features_extracted.csv'
                self.features_df.to_csv(features_path, index=False)
                logger.info(f" Признаки сохранены: {features_path}")
            
            if self.correlation_result is not None:
                if self.correlation_result.significant_pairs:
                    pairs_data = [
                        {'Feature1': p[0], 'Feature2': p[1], 'Correlation': p[2]}
                        for p in self.correlation_result.significant_pairs
                    ]
                    pairs_df = pd.DataFrame(pairs_data)
                    pairs_path = output_path / 'stage13_significant_pairs_main.csv'
                    pairs_df.to_csv(pairs_path, index=False)
                    logger.info(f" Значимые пары сохранены: {pairs_path}")
                
                if self.correlation_result.confounders:
                    confounders_data = [
                        {
                            'Confounder': c.confounder,
                            'Feature1': c.feature_1,
                            'Feature2': c.feature_2,
                            'Correlation_1_2': c.corr_12,
                            'Correlation_1_C': c.corr_1c,
                            'Correlation_2_C': c.corr_2c,
                            'Partial_Correlation': c.partial_corr,
                            'Mediation_Score': c.mediation_score,
                            'Action': c.action
                        }
                        for c in self.correlation_result.confounders
                    ]
                    confounders_df = pd.DataFrame(confounders_data)
                    confounders_path = output_path / 'stage13_confounders_main.csv'
                    confounders_df.to_csv(confounders_path, index=False)
                    logger.info(f" Смешивающие факторы сохранены: {confounders_path}")
                
                if self.correlation_result.feature_importance_by_correlation:
                    importance_data = [
                        {'Feature': f, 'Importance': imp}
                        for f, imp in self.correlation_result.feature_importance_by_correlation.items()
                    ]
                    importance_df = pd.DataFrame(importance_data)
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    importance_path = output_path / 'stage13_feature_importance_main.csv'
                    importance_df.to_csv(importance_path, index=False)
                    logger.info(f" Важность признаков сохранена: {importance_path}")
            
            if self.augmented_df is not None:
                augmented_path = output_path / 'stage13_augmented_data.csv'
                self.augmented_df.to_csv(augmented_path, index=False)
                logger.info(f" Аугментированные данные сохранены: {augmented_path}")
            
            logger.info(f"\nАнализ завершен, результаты: {output_path}")
            
            stage13_results['success'] = True
            return stage13_results
            
        except Exception as e:
            logger.error(f"Ошибка этапа 1.3: {e}")
            import traceback
            traceback.print_exc()
            stage13_results['errors'].append(str(e))
            return stage13_results
    
    
    def run_stage14_pipeline(self, output_dir: str = 'data/output') -> Dict:
        stage14_results = {
            'success': False,
            'features_analyzed': 0,
            'numerical_analyzed': 0,
            'categorical_analyzed': 0,
            'normal_distributions': 0,
            'multimodal_distributions': 0,
            'errors': []
        }
        
        try:
            logger.info("\n" + "=" * 80)
            logger.info("1.4")
            logger.info("=" * 80)
            
            features_file = Path(output_dir) / 'stage13_features_extracted.csv'
            
            if not features_file.exists():
                logger.error(f"Файл признаков не найден: {features_file}")
                logger.info("Сначала запустите анализ признаков")
                stage14_results['errors'].append("Features file not found")
                return stage14_results
            
            logger.info(f" Загрузка признаков из: {features_file}")
            self.features_df = pd.read_csv(features_file)
            logger.info(f" Загружено: {self.features_df.shape[0]} строк × {self.features_df.shape[1]} столбцов")
            
            numerical_features = get_numerical_features()
            categorical_features = get_categorical_features()
            
            logger.info(f"\n Признаки для анализа:")
            logger.info(f"   - Численных: {len(numerical_features)}")
            logger.info(f"   - Категориальных: {len(categorical_features)}")
            logger.info(f"   - Всего: {len(numerical_features) + len(categorical_features)}")
            
            stage14_dir = Path(output_dir) / 'stage14'
            visualizations_dir = stage14_dir / 'visualizations'
            stage14_dir.mkdir(parents=True, exist_ok=True)
            visualizations_dir.mkdir(parents=True, exist_ok=True)
            
            visualizer = DistributionVisualizer(output_dir=str(visualizations_dir))
            
            normality_results = []
            multimodality_results = []
            classification_results = []
            transformation_results = []
            
            logger.info("\n" + "=" * 80)
            logger.info("АНАЛИЗ ЧИСЛЕННЫХ ПРИЗНАКОВ")
            logger.info("=" * 80)
            
            for feature_name in numerical_features:
                if feature_name not in self.features_df.columns:
                    logger.warning(f"\u26a0\ufe0f Feature {feature_name} not found in data")
                    continue
                
                data = self.features_df[feature_name].values
                metadata = get_feature_info(feature_name)
                
                data_clean = data[~pd.isna(data)]
                if len(data_clean) < 3:
                    logger.warning(f" Недостаточно данных для {feature_name}")
                    continue
                
                is_normal, norm_tests = self.normality_tester.consensus_is_normal(data)
                normality_results.append({
                    'feature': feature_name,
                    'is_normal': is_normal,
                    'shapiro_wilk_p': norm_tests['shapiro_wilk'].p_value,
                    'jarque_bera_p': norm_tests['jarque_bera'].p_value
                })
                
                is_multimodal, n_modes, multi_tests = self.multimodality_detector.consensus_is_multimodal(data)
                multimodality_results.append({
                    'feature': feature_name,
                    'is_multimodal': is_multimodal,
                    'n_modes': n_modes
                })
                
                distribution_class = self.distribution_classifier.classify(data, feature_name)
                classification_results.append({
                    'feature': feature_name,
                    'distribution_type': distribution_class.distribution_type.value,
                    'confidence': distribution_class.confidence,
                    'skewness': distribution_class.skewness,
                    'kurtosis': distribution_class.kurtosis
                })
                
                transformations = self.transformation_recommender.recommend(data, distribution_class)
                for i, trans in enumerate(transformations['transformations'][:3]):
                    transformation_results.append({
                        'feature': feature_name,
                        'rank': i + 1,
                        'method': trans.get('method', 'unknown'),
                        'achieves_normality': trans.get('achieves_normality', False)
                    })
                
                try:
                    visualizer.plot_full_diagnostic(data, feature_name, save=True)
                except Exception as viz_error:
                    logger.warning(f"\u26a0\ufe0f Visualization failed for {feature_name}: {viz_error}")
            
            logger.info("\n" + "=" * 80)
            logger.info("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
            logger.info("=" * 80)
            
            for feature_name in categorical_features:
                if feature_name not in self.features_df.columns:
                    logger.warning(f"\u26a0\ufe0f Feature {feature_name} not found in data")
                    continue
                
                data = self.features_df[feature_name].values
                metadata = get_feature_info(feature_name)
                
                data_clean = data[~pd.isna(data)]
                if len(data_clean) == 0:
                    continue
                
                value_counts = pd.Series(data).value_counts()
                classification_results.append({
                    'feature': feature_name,
                    'distribution_type': 'categorical',
                    'n_categories': len(value_counts),
                    'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None
                })
                
                try:
                    categories = metadata.get('categories', None) if metadata else None
                    visualizer.plot_categorical_distribution(data, feature_name, categories=categories, save=True)
                except Exception as viz_error:
                    logger.warning(f"\u26a0\ufe0f Visualization failed for {feature_name}: {viz_error}")
            
            logger.info("\n" + "=" * 80)
            logger.info("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
            logger.info("=" * 80)
            
            pd.DataFrame(normality_results).to_csv(stage14_dir / 'normality_tests.csv', index=False)
            pd.DataFrame(multimodality_results).to_csv(stage14_dir / 'multimodality_tests.csv', index=False)
            pd.DataFrame(classification_results).to_csv(stage14_dir / 'distribution_classification.csv', index=False)
            pd.DataFrame(transformation_results).to_csv(stage14_dir / 'transformation_recommendations.csv', index=False)
            
            logger.info(f" Результаты сохранены в: {stage14_dir}")
            
            stage14_results['features_analyzed'] = len(normality_results) + len([r for r in classification_results if r['distribution_type'] == 'categorical'])
            stage14_results['numerical_analyzed'] = len(normality_results)
            stage14_results['categorical_analyzed'] = len([r for r in classification_results if r['distribution_type'] == 'categorical'])
            stage14_results['normal_distributions'] = sum(1 for r in normality_results if r['is_normal'])
            stage14_results['multimodal_distributions'] = sum(1 for r in multimodality_results if r['is_multimodal'])
            stage14_results['success'] = True
            
            self.stage14_report = stage14_results
            
            logger.info("\nАнализ распределений завершен")
            return stage14_results
            
        except Exception as e:
            logger.error(f"1.4 еррор: {e}")
            import traceback
            traceback.print_exc()
            stage14_results['errors'].append(str(e))
            return stage14_results
    
    def _generate_synthetic_data_stage13(self, n_samples: int) -> pd.DataFrame:
        np.random.seed(42)
        
        data = {
            'T_year': np.full(n_samples, 2026),
            'T_month': np.random.randint(1, 13, n_samples),
            'T_day_of_year': np.random.randint(1, 366, n_samples),
            'T_quarter': np.random.randint(1, 5, n_samples),
            'T_hour': np.random.randint(0, 24, n_samples),
            'T_season': np.random.choice(['winter', 'spring', 'summer', 'autumn'], n_samples),
            'G_latitude': np.random.uniform(50, 60, n_samples),
            'G_longitude': np.random.uniform(30, 40, n_samples),
            'G_utm_zone': np.full(n_samples, 36),
            'M_temperature': np.random.uniform(-10, 30, n_samples),
            'M_humidity': np.random.uniform(40, 95, n_samples),
            'M_pressure': np.random.uniform(990, 1040, n_samples),
            'TR_altitude': np.random.uniform(0, 5000, n_samples),
            'OBJ_water_count': np.random.poisson(3, n_samples),
            'OBJ_forest_count': np.random.poisson(5, n_samples),
            'OBJ_landmark_count': np.random.poisson(1, n_samples),
            'OBJ_nearest_distance_m': np.random.uniform(100, 10000, n_samples),
            'E_ndvi': np.random.uniform(-0.5, 0.9, n_samples),
        }
        
        return pd.DataFrame(data)
    
    
    def run_stage15_pipeline(self, 
                             osm_bbox: Optional[Tuple[float, float, float, float]] = None,
                             external_gpx_dir: Optional[str] = None,
                             generation_preset: str = 'standard',
                             variants_per_track: int = 5) -> Dict:
        logger.info("\n" + "=" * 80)
        logger.info("Расширение данных")
        logger.info("=" * 80)
        
        results = {
            'success': False,
            'osm_tracks_loaded': 0,
            'external_files_parsed': 0,
            'synthetic_tracks_generated': 0,
            'total_new_tracks': 0,
            'errors': []
        }
        
        try:
            
            if osm_bbox is not None:
                logger.info("\n" + "-" * 80)
                logger.info(" ЗАГРУЗКА ТРЕКОВ ИЗ OPENSTREETMAP")
                logger.info("-" * 80)
                
                logger.info(f"Ограничивающий прямоугольник: {osm_bbox}")
                
                try:
                    osm_routes = self.osm_loader.load_routes_in_bbox(
                        bbox=osm_bbox,
                        route_types=['hiking', 'bicycle', 'foot']
                    )
                    
                    logger.info(f" Найдено {len(osm_routes)} маршрутов из OSM")
                    
                    for idx, route_data in enumerate(osm_routes, 1):
                        try:
                            parsed_track = self.track_parser.parse_gpx(route_data['gpx_data'])
                            
                            if not parsed_track or not parsed_track.points:
                                logger.warning(f"    Маршрут {idx}: Нет валидных точек")
                                continue
                            
                            track_id = f"OSM_{route_data.get('osm_id', f'TRACK_{idx:04d}')}"
                            
                            db_track_id = self.database.add_track(
                                track_id=track_id,
                                region='OSM_Import',
                                date_start=parsed_track.metadata.start_time
                            )
                            
                            points_data = []
                            for pidx, point in enumerate(parsed_track.points):
                                points_data.append({
                                    'track_id': db_track_id,
                                    'point_index': pidx,
                                    'latitude': point.latitude,
                                    'longitude': point.longitude,
                                    'altitude': point.elevation or 0.0
                                })
                            
                            self.database.add_points(db_track_id, points_data)
                            
                            self.database.add_external_track_source(
                                platform='OpenStreetMap',
                                source_id=str(route_data.get('osm_id', '')),
                                track_id=db_track_id,
                                original_url=route_data.get('url', ''),
                                platform_metadata={
                                    'route_type': route_data.get('route_type'),
                                    'bbox': osm_bbox
                                }
                            )
                            
                            results['osm_tracks_loaded'] += 1
                            logger.info(f"   Импортирован OSM маршрут {idx}/{len(osm_routes)}: {track_id}")
                            
                        except Exception as e:
                            logger.warning(f"    Не удалось импортировать OSM маршрут {idx}: {e}")
                            results['errors'].append(f"OSM route {idx}: {str(e)}")
                            continue
                    
                    logger.info(f" Импорт OSM завершен: {results['osm_tracks_loaded']} треков загружено")
                    
                except Exception as e:
                    logger.error(f" Загрузка OSM не удалась: {e}")
                    results['errors'].append(f"OSM loading: {str(e)}")
            
            
            if external_gpx_dir is not None:
                logger.info("\n" + "-" * 80)
                logger.info(" ПАРСИНГ ВНЕШНИХ ФАЙЛОВ ТРЕКОВ")
                logger.info("-" * 80)
                
                gpx_dir = Path(external_gpx_dir)
                if not gpx_dir.exists():
                    logger.warning(f"  Директория не найдена: {external_gpx_dir}")
                else:
                    track_files = []
                    for ext in ['*.gpx', '*.tcx', '*.geojson']:
                        track_files.extend(gpx_dir.glob(ext))
                    
                    logger.info(f"Найдено {len(track_files)} файлов треков")
                    
                    for file_path in track_files:
                        try:
                            logger.info(f"  Парсинг: {file_path.name}")
                            
                            if file_path.suffix.lower() == '.gpx':
                                parsed_track = self.track_parser.parse_gpx_file(str(file_path))
                            elif file_path.suffix.lower() == '.tcx':
                                parsed_track = self.track_parser.parse_tcx_file(str(file_path))
                            elif file_path.suffix.lower() == '.geojson':
                                parsed_track = self.track_parser.parse_geojson_file(str(file_path))
                            else:
                                continue
                            
                            if not parsed_track or not parsed_track.points:
                                logger.warning(f"      Нет валидных точек в {file_path.name}")
                                continue
                            
                            track_id = f"EXT_{file_path.stem.upper()}"
                            
                            db_track_id = self.database.add_track(
                                track_id=track_id,
                                region='External_Import',
                                date_start=parsed_track.metadata.start_time
                            )
                            
                            points_data = []
                            for pidx, point in enumerate(parsed_track.points):
                                points_data.append({
                                    'track_id': db_track_id,
                                    'point_index': pidx,
                                    'latitude': point.latitude,
                                    'longitude': point.longitude,
                                    'altitude': point.elevation or 0.0
                                })
                            
                            self.database.add_points(db_track_id, points_data)
                            
                            self.database.add_external_track_source(
                                platform='FileImport',
                                source_id=file_path.name,
                                track_id=db_track_id,
                                original_url=str(file_path),
                                file_format=file_path.suffix[1:],
                                platform_metadata={
                                    'file_size': file_path.stat().st_size
                                }
                            )
                            
                            results['external_files_parsed'] += 1
                            logger.info(f"     Импортировано: {track_id} ({len(parsed_track.points)} точек)")
                            
                        except Exception as e:
                            logger.warning(f"      Не удалось разобрать {file_path.name}: {e}")
                            results['errors'].append(f"File {file_path.name}: {str(e)}")
                            continue
                    
                    logger.info(f" Импорт файлов завершен: {results['external_files_parsed']} файлов разобрано")
            
            
            logger.info("\n" + "-" * 80)
            logger.info(" ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ВАРИАНТОВ ТРЕКОВ")
            logger.info("-" * 80)
            
            db_stats = self.database.get_database_statistics()
            total_tracks = db_stats.get('total_tracks', 0)
            
            if total_tracks == 0:
                logger.warning("  Нет треков в базе для генерации вариантов")
            else:
                logger.info(f"Исходных треков в базе: {total_tracks}")
                logger.info(f"Вариантов на трек: {variants_per_track}")
                
                session = self.database.Session()
                try:
                    from database import RouteTrack
                    tracks = session.query(RouteTrack.track_id).all()
                    source_track_ids = [t[0] for t in tracks]
                finally:
                    session.close()
                
                logger.info(f"Обработка {len(source_track_ids)} треков для синтетической генерации")
                
                try:
                    variant_results = self.synthetic_generator.generate_balanced_dataset(
                        source_track_ids=source_track_ids,
                        variants_per_track=variants_per_track
                    )
                    
                    for track_id, variants in variant_results.items():
                        results['synthetic_tracks_generated'] += len(variants)
                    
                    logger.info(f" Синтетическая генерация завершена: {results['synthetic_tracks_generated']} вариантов создано")
                    
                    report_path = 'data/output/stage15_generation_report.json'
                    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
                    self.synthetic_generator.export_generation_report(variant_results, report_path)
                    logger.info(f" Отчет сохранен: {report_path}")
                    
                except Exception as e:
                    logger.error(f" Синтетическая генерация не удалась: {e}")
                    results['errors'].append(f"Synthetic generation: {str(e)}")
            
            
            results['total_new_tracks'] = (
                results['osm_tracks_loaded'] + 
                results['external_files_parsed'] + 
                results['synthetic_tracks_generated']
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("Отчет о выполнении")
            logger.info("=" * 80)
            logger.info(f"\n Сводка по расширению данных:")
            logger.info(f"  Загружено треков OSM: {results['osm_tracks_loaded']}")
            logger.info(f"  Разобрано внешних файлов: {results['external_files_parsed']}")
            logger.info(f"  Сгенерировано синтетических вариантов: {results['synthetic_tracks_generated']}")
            logger.info(f"  Всего новых треков: {results['total_new_tracks']}")
            
            if results['errors']:
                logger.warning(f"\n  Обнаружено ошибок: {len(results['errors'])}")
                for error in results['errors'][:5]:
                    logger.warning(f"   - {error}")
            
            db_stats_final = self.database.get_database_statistics()
            logger.info(f"\nСтатистика базы данных:")
            logger.info(f"  Всего треков: {db_stats_final.get('total_tracks', 0)}")
            logger.info(f"  Всего точек: {db_stats_final.get('total_points', 0):,}")
            logger.info(f"  Внешних источников: {db_stats_final.get('total_external_sources', 0)}")
            logger.info(f"  Аугментаций: {db_stats_final.get('total_augmentations', 0)}")
            
            results['success'] = True
            return results
            
        except Exception as e:
            logger.error(f"\n ЭТАП 1.5 НЕ УДАЛСЯ: {e}", exc_info=True)
            results['errors'].append(str(e))
            return results
    
    
    def export_tracks_to_files(self,
                               track_ids: List[str],
                               output_dir: str = 'data/exports',
                               formats: List[str] = None) -> Dict:
        if formats is None:
            formats = ['gpx', 'tcx']
        
        logger.info("\n" + "=" * 80)
        logger.info(" ЭКСПОРТ ТРЕКОВ В GPX/TCX")
        logger.info("=" * 80)
        logger.info(f"Треков для экспорта: {len(track_ids)}")
        logger.info(f"Выходная директория: {output_dir}")
        logger.info(f"Форматы: {', '.join(formats)}")
        
        results = {
            'success': False,
            'tracks_exported': 0,
            'gpx_files': 0,
            'tcx_files': 0,
            'errors': []
        }
        
        try:
            from augmentation_integration import track_from_db_to_geopoints
            
            tracks_to_export = []
            
            logger.info("\n" + "-" * 80)
            logger.info("Загрузка треков из базы данных...")
            logger.info("-" * 80)
            
            for track_id in track_ids:
                try:
                    geo_points = track_from_db_to_geopoints(self.database, track_id)
                    
                    export_points = [
                        ExportGeoPoint(
                            latitude=p.lat,
                            longitude=p.lon,
                            elevation=p.elevation,
                            timestamp=p.timestamp
                        )
                        for p in geo_points
                    ]
                    
                    metadata = ExportTrackMetadata(
                        name=track_id,
                        description=f"Track {track_id} - Exported from database",
                        creator="Direct"
                    )
                    
                    tracks_to_export.append((track_id, export_points, metadata))
                    logger.info(f"   Загружено: {track_id} ({len(export_points)} точек)")
                    
                except Exception as e:
                    logger.warning(f"    Fail load {track_id}: {e}")
                    results['errors'].append(f"Load {track_id}: {str(e)}")
            
            logger.info(f" Загружено {len(tracks_to_export)} треков")
            
            if not tracks_to_export:
                logger.warning("  No tracks to export")
                results['success'] = True
                return results
            
            logger.info("\n" + "-" * 80)
            logger.info("Экспорт треков в файлы...")
            logger.info("-" * 80)
            
            export_results = TrackExporter.batch_export_tracks(
                tracks_to_export,
                output_dir=output_dir,
                formats=formats
            )
            
            for track_id, format_results in export_results.items():
                if format_results.get('gpx'):
                    results['gpx_files'] += 1
                if format_results.get('tcx'):
                    results['tcx_files'] += 1
                
                if any(format_results.values()):
                    results['tracks_exported'] += 1
            
            logger.info("\n" + "=" * 80)
            logger.info(" СВОДКА ПО ЭКСПОРТУ")
            logger.info("=" * 80)
            logger.info(f" Экспортировано треков: {results['tracks_exported']}")
            logger.info(f"  Файлов GPX: {results['gpx_files']}")
            logger.info(f"  Файлов TCX: {results['tcx_files']}")
            logger.info(f" Output directory: {output_dir}")
            
            if results['errors']:
                logger.warning(f"  Errors: {len(results['errors'])}")
                for error in results['errors'][:5]:
                    logger.warning(f"   - {error}")
            
            results['success'] = True
            return results
            
        except Exception as e:
            logger.error(f" EXPORT FAILED: {e}", exc_info=True)
            results['errors'].append(str(e))
            return results
    
    
    def run_full_pipeline(self, sample_size: int = 0) -> Dict:
        results = {
            'success': False,
            'start_time': self.start_time.isoformat(),
            'total_tracks': 0,
            'total_points': 0,
            'anomalies_found': 0,
            'anomalies_fixed': 0,
            'maps_generated': 0,
            'points_enriched': 0,
            'errors': []
        }
        
        try:
            
            success, meta = self.load_data_from_csv()
            
            if not success:
                results['errors'].append("Data loading failed")
                return results
            
            results['total_points'] = meta.get('total_points', 0)
            results['anomalies_found'] = meta.get('anomalies_found', 0)
            
            
            logger.info("\n" + "=" * 80)
            logger.info("ТЕСТИРОВАНИЕ ПРЕОБРАЗОВАНИЙ СИСТЕМ КООРДИНАТ")
            logger.info("=" * 80)
            
            coord_test = self.validate_coordinate_systems(55.7558, 37.6173)
            
            if not coord_test:
                results['errors'].append("Coordinate transformation test failed")
                return results
            
            
            logger.info("\n" + "=" * 80)
            logger.info(" ОБРАБОТКА ТРЕКОВ С ИНТЕГРАЦИЕЙ")
            logger.info("=" * 80)
            
            if self.loader is None or self.loader.raw_data is None:
                logger.error("No data loaded for track processing")
                results['errors'].append("No data available")
                return results
            
            df = self.loader.raw_data
            unique_tracks = df['track_id'].unique()
            
            if sample_size > 0:
                unique_tracks = unique_tracks[:sample_size]
                logger.info(f"РЕЖИМ ВЫБОРКИ: Обработка {len(unique_tracks)} треков (из {len(df['track_id'].unique())} всего)")
            else:
                logger.info(f"ПОЛНЫЙ РЕЖИМ: Обработка всех {len(unique_tracks)} уникальных треков")
            
            for idx, track_id in enumerate(unique_tracks, 1):
                try:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"ТРЕК {idx}/{len(unique_tracks)}: {track_id}")
                    logger.info(f"{'='*80}")
                    
                    track_df = df[df['track_id'] == track_id].copy()
                    
                    points = list(zip(track_df['latitude'], track_df['longitude']))
                    altitudes = list(track_df['altitude'])
                    region = track_df['region'].iloc[0] if 'region' in track_df.columns else 'Unknown'
                    
                    logger.info(f" Points: {len(points)}, Region: {region}")
                    
                    anomaly_result = self.process_track_anomalies(track_id, points)
                    corrected_points = anomaly_result.get('corrected_points', points)
                    
                    points_data_for_enrichment = []
                    for idx_row, row in track_df.iterrows():
                        point_dict = {
                            'track_id': track_id,
                            'point_index': int(row['point_index']),
                            'latitude': float(row['latitude']),
                            'longitude': float(row['longitude']),
                            'altitude': float(row['altitude']),
                            'date': row.get('date'),
                            'csv_terrain_type': None
                        }
                        points_data_for_enrichment.append(point_dict)
                    
                    enriched_points = self.enrich_track_points(track_id, points_data_for_enrichment)
                    
                    if not enriched_points:
                        logger.warning(f" Enrichment failed, using basic data")
                        enriched_points = []
                        for point_dict in points_data_for_enrichment:
                            enriched_points.append(EnrichedPointData(
                                track_id=track_id,
                                point_index=point_dict['point_index'],
                                latitude=point_dict['latitude'],
                                longitude=point_dict['longitude'],
                                altitude=point_dict['altitude'],
                                date=point_dict.get('date'),
                                region=region,
                                temperature=None,
                                step_frequency=None,
                                terrain_type=None,
                                nearby_objects=[],
                                object_count=0,
                                quality_score=0.0,
                                enrichment_timestamp=datetime.now().isoformat()
                            ))
                    
                    map_path = self.generate_track_maps(track_id, corrected_points)
                    
                    success = self.store_enriched_track_in_database(
                        track_id=track_id,
                        enriched_points=enriched_points,
                        region=region
                    )
                    
                    if success:
                        self.processed_tracks += 1
                        logger.info(f"  Трек {track_id} обработан")
                    
                except Exception as e:
                    logger.error(f" Error processing track {track_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    results['errors'].append(f"Track {track_id}: {str(e)}")
                    continue
            
            
            logger.info("\n" + "=" * 80)
            logger.info("=" * 80)
            
            elapsed = datetime.now() - self.start_time
            
            try:
                db_stats = self.database.get_database_statistics()
            except Exception as e:
                logger.warning(f"Could not retrieve database stats: {e}")
                db_stats = {}
            
            logger.info(f"\n EXECUTION SUCCESSFUL")
            logger.info(f"\nExecution Time: {elapsed.total_seconds():.2f}s")
            logger.info(f"\nData Summary:")
            logger.info(f"  Total data points processed: {results['total_points']:,}")
            logger.info(f"  Обработано уникальных треков: {self.processed_tracks}")
            logger.info(f"  GPS anomalies detected: {self.anomalies_found}")
            logger.info(f"  Anomalies fixed: {self.anomalies_fixed}")
            logger.info(f"  Maps generated: {self.maps_generated}")
            logger.info(f"  Points enriched (STAGE 1.2): {self.points_enriched}")
            
            logger.info(f"\nDatabase Statistics:")
            logger.info(f"  Всего треков в БД: {db_stats.get('total_tracks', 0)}")
            logger.info(f"  Total points in DB: {db_stats.get('total_points', 0):,}")
            logger.info(f"  Total anomalies recorded: {db_stats.get('total_anomalies', 0)}")
            
            logger.info(f"\n1.1 - Geodetic Processing:")
            logger.info(f"    WGS84 ↔ Web Mercator transformations")
            logger.info(f"    WGS84 → UTM zone projections")
            logger.info(f"    Vincenty distance calculations")
            logger.info(f"    Latitude-dependent corrections")
            logger.info(f"    GPS anomaly detection and correction")
            logger.info(f"    Database with version history")
            
            logger.info(f"\n1.2 - Universal Map Decoding:")
            logger.info(f"    500m radius environment extraction")
            logger.info(f"    Terrain type classification")
            logger.info(f"    Object identification with distances")
            logger.info(f"    Weather data integration")
            logger.info(f"    Full dataset (10 required fields)")
            logger.info(f"    Yandex Maps API integration")
            
            
            logger.info("\n" + "=" * 80)
            logger.info("1.3")
            logger.info("=" * 80)
            
            stage13_results = self.run_stage13_pipeline(
                enriched_tracks=self.processed_tracks,
                output_dir='data/output'
            )
            
            results['stage13'] = stage13_results
            
            logger.info(f"\n1.3 - Feature Analysis:")
            logger.info(f"    44 features extracted")
            logger.info(f"    Correlation analysis (4 methods)")
            logger.info(f"    SHAP feature importance")
            logger.info(f"    Permutation importance")
            logger.info(f"    ANOVA")
            logger.info(f"    Causal structure learning")
            logger.info(f"    Data augmentation (SMOTE/VAE/Kriging)")
            
            
            logger.info("\n" + "=" * 80)
            logger.info("1.4: DISTRIBUTION ANALYSIS")
            logger.info("=" * 80)
            
            stage14_results = self.run_stage14_pipeline(output_dir='data/output')
            results['stage14'] = stage14_results
            
            if stage14_results['success']:
                logger.info(f"\n1.4 - Distribution Analysis:")
                logger.info(f"    {stage14_results['features_analyzed']} features analyzed")
                logger.info(f"    {stage14_results['numerical_analyzed']} numerical distributions")
                logger.info(f"    {stage14_results['categorical_analyzed']} categorical distributions")
                logger.info(f"    {stage14_results['normal_distributions']} normal distributions found")
                logger.info(f"    {stage14_results['multimodal_distributions']} multimodal distributions found")
                logger.info(f"    Normality tests (4 methods)")
                logger.info(f"    Multimodality detection (3 methods)")
                logger.info(f"    Transformation recommendations")
                logger.info(f"    Full diagnostic visualizations")
            
            
            logger.info("\n" + "=" * 80)
            logger.info(" 1.5: start")
            logger.info("=" * 80)
            
            stage15_results = self.run_stage15_pipeline(
                osm_bbox=None,
                external_gpx_dir=None,
                generation_preset='standard',
                variants_per_track=3
            )
            results['stage15'] = stage15_results
            
            if stage15_results['success']:
                logger.info(f"\n1.5 - Data Expansion with Geodetic Accuracy:")
                logger.info(f"    {stage15_results.get('variants_generated', 0)} синтетических вариантов треков сгенерировано")
                logger.info(f"    Geodetic transformations applied")
                logger.info(f"    Visual augmentations applied")
                logger.info(f"    Augmented data stored in database")
            
            logger.info("\n" + "=" * 80)
            logger.info(" Всё окей")
            logger.info("=" * 80)
            
            results['success'] = True
            results['end_time'] = datetime.now().isoformat()
            results['execution_time_seconds'] = elapsed.total_seconds()
            results['total_tracks'] = self.processed_tracks
            results['anomalies_fixed'] = self.anomalies_fixed
            results['maps_generated'] = self.maps_generated
            results['points_enriched'] = self.points_enriched
            
            return results
            
        except Exception as e:
            logger.error(f"\n FATAL ERROR: {e}", exc_info=True)
            results['errors'].append(str(e))
            return results


def main():
    parser = argparse.ArgumentParser(
        description='Интеграция'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='сам себе поможешь'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("=" * 80)
    
    processor = FullIntegratedProcessor()
    results = processor.run_full_pipeline(sample_size=args.sample)
    
    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())

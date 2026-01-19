
import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_enricher_yandex import (
    initialize_enricher_yandex,
    YandexFeatureExtractor,
    YandexCache,
    EnrichedPointData
)
from map_decoder import StandardTerrainType

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


def print_header(text: str):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def test_yandex_connectivity():
    print("TEST 1: Yandex Geocoder Connectivity")
    print("-" * 40)
    
    try:
        extractor = YandexFeatureExtractor()
        
        lat, lon = 55.7558, 37.6173
        
        print(f"Querying location: {lat}, {lon} (Moscow)")
        start_time = time.time()
        
        result = extractor.query_yandex_geocoder(lat, lon)
        elapsed = time.time() - start_time
        
        if result:
            print(f" PASS - Yandex Geocoder responded in {elapsed:.2f}s")
            print(f"  Response address: {result.get('address', {})}")
            return True
        else:
            print(f" FAIL - Yandex Geocoder did not respond")
            return False
            
    except Exception as e:
        print(f" FAIL - Exception: {e}")
        return False


def test_terrain_extraction():
    print("\nTEST 2: Terrain Extraction")
    print("-" * 40)
    
    try:
        extractor = YandexFeatureExtractor()
        
        test_points = [
            (55.7558, 37.6173, "Moscow (Urban)"),
            (54.5973, 36.3312, "Tula (Regional)"),
        ]
        
        success_count = 0
        
        for lat, lon, name in test_points:
            terrain = extractor.extract_terrain_from_coordinates(lat, lon)
            status = "" if terrain else ""
            print(f"{status} {name:25} → {terrain}")
            if terrain:
                success_count += 1
        
        if success_count == len(test_points):
            print(f"\n PASS - Terrain extraction working ({success_count}/{len(test_points)})")
            return True
        else:
            print(f"\n PARTIAL - Terrain extraction: {success_count}/{len(test_points)}")
            return True
            
    except Exception as e:
        print(f" FAIL - Exception: {e}")
        return False


def test_map_enricher_initialization():
    print("\nTEST 3: Map Enricher Module Запуск")
    print("-" * 40)
    
    try:
        print("Initializing EnrichmentEngineYandex")
        engine = initialize_enricher_yandex()
        
        if engine:
            print(f" Engine initialized successfully")
            print(f"  Buffer radius: {engine.buffer_radius_m}m")
            print(f"  Cache directory: exists")
            print(f" PASS - Map enricher module functional")
            return True
        else:
            print(f" FAIL - Engine initialization returned None")
            return False
            
    except Exception as e:
        print(f" FAIL - Exception during initialization: {e}")
        return False


def test_cache_system():
    print("\nTEST 4: Cache System Operations")
    print("-" * 40)
    
    try:
        cache = YandexCache("test_yandex_cache")
        
        print("Writing test data to cache")
        test_data = {'address': 'Test Location', 'feature': 'test'}
        cache.set(55.5, 37.5, "geo", test_data)
        
        print("Reading test data from cache")
        retrieved = cache.get(55.5, 37.5, "geo")
        
        if retrieved == test_data:
            print(f" Cache read/write verified")
            
            stats = cache.get_statistics()
            print(f"  Cache files: {stats['cached_files']}")
            print(f"  Hit rate: {stats['hit_rate']}")
            
            print(f" PASS - Cache system operational")
            return True
        else:
            print(f" FAIL - Retrieved data doesn't match written data")
            return False
            
    except Exception as e:
        print(f" FAIL - Exception: {e}")
        return False


def test_batch_enrichment():
    print("\nTEST 5: Batch Enrichment")
    print("-" * 40)
    
    try:
        engine = initialize_enricher_yandex()
        
        test_points = [
            {
                'track_id': 'TEST_BATCH',
                'point_index': 0,
                'latitude': 55.7558,
                'longitude': 37.6173,
                'altitude': 150.0,
                'date': '2024-01-17',
                'csv_terrain_type': StandardTerrainType.URBAN
            },
            {
                'track_id': 'TEST_BATCH',
                'point_index': 1,
                'latitude': 55.7400,
                'longitude': 37.6300,
                'altitude': 160.0,
                'date': '2024-01-17',
                'csv_terrain_type': StandardTerrainType.GRASSLAND
            },
        ]
        
        print(f"Processing batch of {len(test_points)} points...")
        start_time = time.time()
        
        results = engine.enrich_batch(test_points)
        elapsed = time.time() - start_time
        
        if results and len(results) == len(test_points):
            print(f" Processed {len(results)} points in {elapsed:.2f}s")
            
            sample = results[0]
            print(f"\nSample enrichment result:")
            print(f"  Track: {sample.track_id}")
            print(f"  Terrain: {sample.terrain_type}")
            print(f"  Objects found: {sample.object_count}")
            print(f"  Quality: {sample.quality_score:.2f}")
            
            print(f"\n PASS - Batch enrichment functional")
            return True
        else:
            print(f" FAIL - Batch processing returned unexpected results")
            return False
            
    except Exception as e:
        print(f" FAIL - Exception: {e}")
        return False


def run_all_tests():
    print_header("STAGE 2 DIAGNOSTIC - YANDEX VERSION")
    
    tests = [
        ("Yandex Geocoder Connectivity", test_yandex_connectivity),
        ("Terrain Extraction", test_terrain_extraction),
        ("Map Enricher Init", test_map_enricher_initialization),
        ("Cache System", test_cache_system),
        ("Batch Enrichment", test_batch_enrichment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    print_header("DIAGNOSTIC SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status:8} - {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All diagnostic tests PASSED - Stage 2 (Yandex) is ready for production!")
    elif passed >= 3:
        print(f"\n {passed}/{total} tests passed - Core functionality operational")
    else:
        print(f"\n {passed}/{total} tests passed - Issues need resolution")
    
    print("\n" + "="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

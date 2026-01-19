
import sys
import logging
from pathlib import Path
import requests
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_overpass_connectivity():
    print("\n" + "="*80)
    print("TEST 1: Overpass API Connectivity")
    print("="*80)
    
    url = "https://overpass-api.de/api/interpreter"
    
    print(f"\nTesting connection to: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Response length: {len(response.text)} bytes")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        
        if response.status_code != 200:
            print(f"  WARNING: Unexpected status code {response.status_code}")
            print(f"  Response text: {response.text[:200]}")
            return False
        
        print(" Connectivity OK")
        return True
        
    except requests.Timeout:
        print(" TIMEOUT: Server not responding within 10 seconds")
        return False
    except requests.ConnectionError as e:
        print(f"   CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_overpass_query():
    print("\n" + "="*80)
    print("TEST 2: Overpass Query Execution")
    print("="*80)
    
    latitude = 55.7558
    longitude = 37.6173
    radius_m = 500
    
    radius_deg = radius_m / 111000.0
    
    south = latitude - radius_deg
    west = longitude - radius_deg
    north = latitude + radius_deg
    east = longitude + radius_deg
    
    query = f"""[out:json];
[bbox:{south},{west},{north},{east}];
(
  node["natural"="peak"];
  node["natural"="spring"];
);
out geom;"""
    
    print(f"\nQuery parameters:")
    print(f"  Location: Moscow ({latitude}, {longitude})")
    print(f"  Radius: {radius_m}m")
    print(f"  Bounding box: [{south}, {west}, {north}, {east}]")
    print(f"  Query length: {len(query)} characters")
    
    try:
        print("\nSending query")
        start = time.time()
        
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            params={'data': query},
            timeout=30,
            headers={'User-Agent': 'DiagnosticTest/1.0'}
        )
        
        elapsed = time.time() - start
        
        print(f"  Status: {response.status_code}")
        print(f"  Response time: {elapsed:.2f}s")
        print(f"  Response length: {len(response.text)} bytes")
        
        if response.status_code != 200:
            print(f"   Non-200 status code")
            print(f"  Response: {response.text[:200]}")
            return False
        
        if not response.text or response.text.strip() == '':
            print(f"   Empty response")
            return False
        
        try:
            data = response.json()
            print(f"   Valid JSON response")
            print(f"  Elements found: {len(data.get('elements', []))}")
            return True
        except Exception as json_err:
            print(f"   Invalid JSON: {json_err}")
            print(f"  Response (first 300 chars): {response.text[:300]}")
            return False
            
    except requests.Timeout:
        print(f"   TIMEOUT: Query took too long")
        return False
    except requests.ConnectionError as e:
        print(f"   CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_map_enricher():
    print("\n" + "="*80)
    print("TEST 3: Map Enricher Module")
    print("="*80)
    
    try:
        from map_enricher_optimized import initialize_enricher
        from map_decoder import StandardTerrainType
        
        print("\n Initializing enricher")
        engine = initialize_enricher(buffer_radius_m=500)
        print(" Enricher Запуск")
        
        print("\n Testing single point enrichment")
        enriched = engine.enrich_point(
            track_id='TEST_001',
            point_index=0,
            latitude=55.7558,
            longitude=37.6173,
            altitude=150.0,
            date='2024-01-17',
            csv_terrain_type=StandardTerrainType.GRASSLAND
        )
        
        print(f"   Point enriched")
        print(f"    - Terrain: {enriched.terrain_type}")
        print(f"    - Confidence: {enriched.terrain_confidence}")
        print(f"    - Objects found: {enriched.object_count}")
        print(f"    - Quality score: {enriched.quality_score:.2f}")
        print(f"    - OSM available: {enriched.osm_available}")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_system():
    print("\n" + "="*80)
    print("TEST 4: Cache System")
    print("="*80)
    
    try:
        from map_enricher_optimized import OSMCache
        
        cache = OSMCache(cache_dir="osm_cache_diagnostic")
        
        print("\n Testing cache operations")
        
        test_data = {'elements': [], 'version': 1.0}
        cache.set(55.7558, 37.6173, 500, test_data)
        print(" Cache write successful")
        
        retrieved = cache.get(55.7558, 37.6173, 500)
        if retrieved and retrieved == test_data:
            print(" Cache read successful")
        else:
            print(" Cache read")
            return False
        
        stats = cache.get_statistics()
        print(f"\n  Cache statistics:")
        print(f"    - Hits: {stats['hits']}")
        print(f"    - Misses: {stats['misses']}")
        print(f"    - Hit rate: {stats['hit_rate']}")
        print(f"    - Files: {stats['cached_files']}")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rate_limiter():
    print("\n" + "="*80)
    print("TEST 5: Rate Limiter")
    print("="*80)
    
    try:
        from map_enricher_optimized import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=30)
        
        print("\n Testing rate limiter")
        
        for i in range(5):
            limiter.record_request()
            stats = limiter.get_statistics()
        
        print(f"   Rate limiter functional")
        print(f"\n  Limiter statistics:")
        print(f"    - Total requests: {stats['total_requests']}")
        print(f"    - Min interval: {stats['min_interval']}")
        print(f"    - Requests/min: {stats['requests_per_min']}")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    
    print("\n")
    print("" + "="*78 + "")
    print("" + " "*78 + "")
    print("" + " STAGE 2 DIAGNOSTIC TEST - OSM Integration Validation ".center(78) + "")
    print("" + " "*78 + "")
    print("" + "="*78 + "")
    
    results = {
        "Overpass Connectivity": test_overpass_connectivity(),
        "Overpass Query": test_overpass_query(),
        "Map Enricher": test_map_enricher(),
        "Cache System": test_cache_system(),
        "Rate Limiter": test_rate_limiter()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        print(f"  {status:8} {test_name}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All diagnostics passed - Stage 2 is working correctly!")
        return 0
    else:
        print(f"\n   {total - passed} test(s) failed - See details above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

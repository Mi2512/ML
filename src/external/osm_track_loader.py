
import logging
import requests
import time
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class OSMTrackLoader:
    
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    RATE_LIMIT_DELAY = 1.0
    
    def __init__(self, cache_dir: str = "data/cache/osm"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
        self.request_count = 0
        
    def query_tracks_in_bbox(self, 
                            bbox: Tuple[float, float, float, float],
                            max_tracks: int = 100,
                            min_points: int = 10) -> List[Dict]:
        logger.info(f"Querying OSM tracks in bbox: {bbox}")
        
        cache_key = self._get_cache_key(bbox)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f" Loaded {len(cached_data)} tracks from cache")
            return cached_data[:max_tracks]
        
        query = self._build_overpass_query(bbox, max_tracks)
        
        response_data = self._execute_query(query)
        
        if response_data is None:
            return []
        
        tracks = self._parse_overpass_response(response_data, min_points)
        
        self._save_to_cache(cache_key, tracks)
        
        logger.info(f" Retrieved {len(tracks)} tracks from OSM")
        return tracks[:max_tracks]
    
    def _build_overpass_query(self, bbox: Tuple[float, float, float, float], 
                              max_results: int) -> str:
        min_lat, min_lon, max_lat, max_lon = bbox
        
        query = f"""
        [out:json][timeout:30];
        (
          way["highway"="path"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["highway"="footway"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["highway"="track"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["route"="hiking"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom {max_results};
        """
        
        return query.strip()
    
    def _execute_query(self, query: str, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                self._respect_rate_limit()
                
                response = requests.post(
                    self.OVERPASS_URL,
                    data={'data': query},
                    timeout=60
                )
                
                self.request_count += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Query failed: HTTP {response.status_code}")
                    return None
                    
            except Exception as e:
                logger.error(f"Query error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        return None
    
    def _parse_overpass_response(self, data: Dict, min_points: int) -> List[Dict]:
        tracks = []
        
        for element in data.get('elements', []):
            track = None
            
            if element['type'] == 'way':
                track = self._parse_way(element)
            elif element['type'] == 'relation':
                track = self._parse_relation(element)
                
            if track and len(track.get('points', [])) >= min_points:
                tracks.append(track)
                
        return tracks
    
    def _parse_way(self, way: Dict) -> Optional[Dict]:
        if 'geometry' not in way:
            return None
            
        points = []
        for node in way['geometry']:
            points.append({
                'lat': node['lat'],
                'lon': node['lon'],
                'ele': node.get('ele', 0.0)
            })
            
        return {
            'id': f"osm_way_{way['id']}",
            'points': points,
            'tags': way.get('tags', {}),
            'activity_type': self._infer_activity_type(way.get('tags', {})),
            'source': 'OpenStreetMap'
        }
    
    def _parse_relation(self, relation: Dict) -> Optional[Dict]:
        points = []
        
        for member in relation.get('members', []):
            if member['type'] == 'way' and 'geometry' in member:
                for node in member['geometry']:
                    points.append({
                        'lat': node['lat'],
                        'lon': node['lon'],
                        'ele': node.get('ele', 0.0)
                    })
                    
        if not points:
            return None
            
        return {
            'id': f"osm_rel_{relation['id']}",
            'points': points,
            'tags': relation.get('tags', {}),
            'activity_type': self._infer_activity_type(relation.get('tags', {})),
            'source': 'OpenStreetMap'
        }
    
    def _infer_activity_type(self, tags: Dict) -> str:
        if 'route' in tags:
            return tags['route']
        elif 'highway' in tags:
            highway = tags['highway']
            if highway in ['path', 'footway']:
                return 'hiking'
            elif highway == 'cycleway':
                return 'cycling'
        return 'unknown'
    
    def _respect_rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
            
        self.last_request_time = time.time()
    
    def _get_cache_key(self, bbox: Tuple[float, float, float, float]) -> str:
        bbox_str = '_'.join(str(round(x, 4)) for x in bbox)
        return hashlib.md5(bbox_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, tracks: List[Dict]):
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(tracks, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def export_to_gpx(self, track: Dict, output_path: str):
        root = ET.Element('gpx', version="1.1", creator="OSMTrackLoader")
        
        trk = ET.SubElement(root, 'trk')
        ET.SubElement(trk, 'name').text = track.get('id', 'Unknown')
        ET.SubElement(trk, 'type').text = track.get('activity_type', 'unknown')
        
        trkseg = ET.SubElement(trk, 'trkseg')
        
        for point in track['points']:
            trkpt = ET.SubElement(trkseg, 'trkpt', 
                                 lat=str(point['lat']), 
                                 lon=str(point['lon']))
            if point.get('ele'):
                ET.SubElement(trkpt, 'ele').text = str(point['ele'])
                
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
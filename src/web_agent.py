
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import requests
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)


class WebAgent:
    
    ALLTRAILS_API = 'https://www.alltrails.com/api/v1'
    KOMOOT_API = 'https://www.komoot.com/api'
    
    def __init__(self, download_dir: str = 'data/temp/downloads',
                 rate_limit: float = 1.0):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RouteAnalysisBot/1.0 (+http://example.com/bot)'
        })
        
        logger.info(f"WebAgent initialized with download directory: {download_dir}")
    
    def download_gpx_file(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        try:
            time.sleep(self.rate_limit)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if not response.headers.get('content-type', '').startswith('application'):
                if '<gpx' not in response.text[:1000]:
                    logger.warning(f"Downloaded file doesn't appear to be GPX: {url}")
                    return None
            
            if not filename:
                parsed = urlparse(url)
                filename = Path(parsed.path).name or f'track_{datetime.now().timestamp()}.gpx'
            
            filepath = self.download_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded GPX file: {url} -> {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading GPX file: {e}")
            return None
    
    def parse_gpx_file(self, gpx_path: Path) -> Optional[Dict]:
        try:
            tree = ET.parse(gpx_path)
            root = tree.getroot()
            
            namespaces = {
                '': 'http://www.topografix.com/GPX/1/1',
                'gpx': 'http://www.topografix.com/GPX/1/1'
            }
            
            track_data = {
                'name': None,
                'description': None,
                'points': [],
                'segments': [],
                'created_at': datetime.now().isoformat()
            }
            
            for elem in root.iter():
                if 'name' in elem.tag.lower():
                    track_data['name'] = elem.text
                    break
            
            for trkpt in root.iter():
                if 'trkpt' in trkpt.tag.lower():
                    lat = trkpt.get('lat')
                    lon = trkpt.get('lon')
                    
                    if lat and lon:
                        point = {
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'altitude': None,
                            'timestamp': None
                        }
                        
                        for child in trkpt:
                            if 'ele' in child.tag.lower():
                                point['altitude'] = float(child.text or 0)
                            elif 'time' in child.tag.lower():
                                point['timestamp'] = child.text
                        
                        track_data['points'].append(point)
            
            logger.info(f"Parsed GPX file: {track_data['name']} with {len(track_data['points'])} points")
            
            return track_data if track_data['points'] else None
            
        except Exception as e:
            logger.error(f"Error parsing GPX file: {e}")
            return None
    
    def parse_kml_file(self, kml_path: Path) -> Optional[Dict]:
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()
            
            track_data = {
                'name': None,
                'description': None,
                'points': [],
                'created_at': datetime.now().isoformat()
            }
            
            for placemark in root.iter():
                if 'placemark' in placemark.tag.lower():
                    for elem in placemark:
                        if 'name' in elem.tag.lower():
                            track_data['name'] = elem.text
                        elif 'description' in elem.tag.lower():
                            track_data['description'] = elem.text
                    
                    for coords_elem in placemark.iter():
                        if 'coordinates' in coords_elem.tag.lower():
                            coords_text = coords_elem.text
                            if coords_text:
                                for coord in coords_text.strip().split():
                                    parts = coord.split(',')
                                    if len(parts) >= 2:
                                        point = {
                                            'latitude': float(parts[1]),
                                            'longitude': float(parts[0]),
                                            'altitude': float(parts[2]) if len(parts) > 2 else None,
                                            'timestamp': None
                                        }
                                        track_data['points'].append(point)
            
            logger.info(f"Parsed KML file: {track_data['name']} with {len(track_data['points'])} points")
            
            return track_data if track_data['points'] else None
            
        except Exception as e:
            logger.error(f"Error parsing KML file: {e}")
            return None
    
    def fetch_alltrails_data(self, trail_id: str, api_key: Optional[str] = None) -> Optional[Dict]:
        try:
            url = f'{self.ALLTRAILS_API}/trails/{trail_id}'
            
            params = {
                '_format': 'json',
                '_apikey': api_key
            } if api_key else {}
            
            time.sleep(self.rate_limit)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            trail_data = response.json()
            
            logger.info(f"Fetched AllTrails data for trail {trail_id}")
            
            return trail_data
            
        except Exception as e:
            logger.error(f"Error fetching AllTrails data: {e}")
            return None
    
    def batch_download_tracks(self, urls: List[str]) -> List[Path]:
        downloaded = []
        
        for i, url in enumerate(urls):
            logger.info(f"Downloading track {i+1}/{len(urls)}: {url}")
            
            if url.endswith('.gpx') or 'gpx' in url:
                filepath = self.download_gpx_file(url)
            elif url.endswith('.kml') or 'kml' in url:
                filepath = self.download_gpx_file(url)
            else:
                filepath = self.download_gpx_file(url)
            
            if filepath:
                downloaded.append(filepath)
        
        logger.info(f"Successfully downloaded {len(downloaded)}/{len(urls)} tracks")
        
        return downloaded
    
    def export_download_metadata(self, tracks: List[Dict], output_path: str) -> None:
        metadata = {
            'total_tracks': len(tracks),
            'download_date': datetime.now().isoformat(),
            'tracks': [
                {
                    'name': t.get('name'),
                    'point_count': len(t.get('points', [])),
                    'source': t.get('source', 'unknown')
                }
                for t in tracks
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported download metadata to {output_path}")


if __name__ == "__main__":
    agent = WebAgent()
    
    url = "https://example.com/track.gpx"
    filepath = agent.download_gpx_file(url, "example_track.gpx")
    
    if filepath:
        track_data = agent.parse_gpx_file(filepath)
        print(f"Parsed track: {track_data}")

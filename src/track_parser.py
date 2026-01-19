
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone
import json
import math
from pathlib import Path

logger = logging.getLogger(__name__)



@dataclass
class TrackPoint:
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude} (must be -90 to 90)")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude} (must be -180 to 180)")
        if self.elevation is not None and (self.elevation < -500 or self.elevation > 9000):
            logger.warning(f"Suspicious elevation: {self.elevation}m (expected -500 to 9000m)")


@dataclass
class TrackMetadata:
    name: Optional[str] = None
    description: Optional[str] = None
    activity_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source: Optional[str] = None
    creator: Optional[str] = None
    total_distance_m: Optional[float] = None
    elevation_gain_m: Optional[float] = None
    elevation_loss_m: Optional[float] = None
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None
    duration_seconds: Optional[float] = None
    avg_speed_kmh: Optional[float] = None


@dataclass
class ParsedTrack:
    points: List[TrackPoint] = field(default_factory=list)
    metadata: TrackMetadata = field(default_factory=TrackMetadata)
    format: str = "unknown"
    parse_errors: List[str] = field(default_factory=list)
    
    def __len__(self):
        return len(self.points)
    
    def is_valid(self) -> bool:
        return len(self.points) >= 2 and len(self.parse_errors) == 0



class GPXParser:
    
    NS_GPX_10 = "{http://www.topografix.com/GPX/1/0}"
    NS_GPX_11 = "{http://www.topografix.com/GPX/1/1}"
    
    @staticmethod
    def detect_namespace(root: ET.Element) -> str:
        if GPXParser.NS_GPX_11 in root.tag:
            return GPXParser.NS_GPX_11
        elif GPXParser.NS_GPX_10 in root.tag:
            return GPXParser.NS_GPX_10
        else:
            return ""
    
    @staticmethod
    def parse(file_path: str) -> ParsedTrack:
        result = ParsedTrack(format="gpx")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = GPXParser.detect_namespace(root)
            
            result.metadata = GPXParser._extract_metadata(root, ns)
            
            for trk in root.findall(f"{ns}trk"):
                for trkseg in trk.findall(f"{ns}trkseg"):
                    for trkpt in trkseg.findall(f"{ns}trkpt"):
                        try:
                            point = GPXParser._parse_trackpoint(trkpt, ns)
                            result.points.append(point)
                        except Exception as e:
                            error_msg = f"Failed to parse trackpoint: {e}"
                            result.parse_errors.append(error_msg)
                            logger.debug(error_msg)
            
            if len(result.points) >= 2:
                GPXParser._calculate_statistics(result)
            
            logger.info(f" Parsed GPX: {len(result.points)} points, {len(result.parse_errors)} errors")
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error: {e}"
            result.parse_errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing GPX: {e}"
            result.parse_errors.append(error_msg)
            logger.error(error_msg)
        
        return result
    
    @staticmethod
    def _extract_metadata(root: ET.Element, ns: str) -> TrackMetadata:
        metadata = TrackMetadata()
        
        metadata.creator = root.get('creator')
        
        trk = root.find(f"{ns}trk")
        if trk is not None:
            name_elem = trk.find(f"{ns}name")
            if name_elem is not None:
                metadata.name = name_elem.text
            
            desc_elem = trk.find(f"{ns}desc")
            if desc_elem is not None:
                metadata.description = desc_elem.text
            
            type_elem = trk.find(f"{ns}type")
            if type_elem is not None:
                metadata.activity_type = type_elem.text
        
        meta = root.find(f"{ns}metadata")
        if meta is not None:
            time_elem = meta.find(f"{ns}time")
            if time_elem is not None:
                metadata.start_time = GPXParser._parse_time(time_elem.text)
        
        return metadata
    
    @staticmethod
    def _parse_trackpoint(trkpt: ET.Element, ns: str) -> TrackPoint:
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        
        ele_elem = trkpt.find(f"{ns}ele")
        elevation = float(ele_elem.text) if ele_elem is not None and ele_elem.text else None
        
        time_elem = trkpt.find(f"{ns}time")
        timestamp = GPXParser._parse_time(time_elem.text) if time_elem is not None and time_elem.text else None
        
        return TrackPoint(
            latitude=lat,
            longitude=lon,
            elevation=elevation,
            timestamp=timestamp
        )
    
    @staticmethod
    def _parse_time(time_str: str) -> Optional[datetime]:
        if not time_str:
            return None
        
        try:
            for fmt in [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z"
            ]:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    if dt.tzinfo is not None:
                        return dt.astimezone(timezone.utc).replace(tzinfo=None)
                    return dt
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse time: {time_str}")
            return None
            
        except Exception as e:
            logger.warning(f"Time parsing error: {e}")
            return None
    
    @staticmethod
    def _calculate_statistics(track: ParsedTrack):
        points = track.points
        metadata = track.metadata
        
        timestamps = [p.timestamp for p in points if p.timestamp is not None]
        if timestamps:
            metadata.start_time = min(timestamps)
            metadata.end_time = max(timestamps)
            metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
        
        elevations = [p.elevation for p in points if p.elevation is not None]
        if elevations:
            metadata.min_elevation_m = min(elevations)
            metadata.max_elevation_m = max(elevations)
            
            gain = 0.0
            loss = 0.0
            for i in range(1, len(elevations)):
                diff = elevations[i] - elevations[i-1]
                if diff > 0:
                    gain += diff
                else:
                    loss += abs(diff)
            
            metadata.elevation_gain_m = round(gain, 2)
            metadata.elevation_loss_m = round(loss, 2)
        
        total_distance = 0.0
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            dist = GPXParser._haversine(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
            total_distance += dist
        
        metadata.total_distance_m = round(total_distance, 2)
        
        if metadata.duration_seconds and metadata.duration_seconds > 0:
            metadata.avg_speed_kmh = round((metadata.total_distance_m / 1000) / (metadata.duration_seconds / 3600), 2)
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c



class TCXParser:
    
    NS_TCX = "{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
    
    @staticmethod
    def parse(file_path: str) -> ParsedTrack:
        result = ParsedTrack(format="tcx")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = TCXParser.NS_TCX
            
            activities = root.find(f"{ns}Activities")
            if activities is None:
                result.parse_errors.append("No Activities element found in TCX")
                return result
            
            for activity in activities.findall(f"{ns}Activity"):
                activity_type = activity.get('Sport', 'Unknown')
                if not result.metadata.activity_type:
                    result.metadata.activity_type = activity_type
                
                for lap in activity.findall(f"{ns}Lap"):
                    lap_start = lap.get('StartTime')
                    if lap_start and not result.metadata.start_time:
                        result.metadata.start_time = GPXParser._parse_time(lap_start)
                    
                    track = lap.find(f"{ns}Track")
                    if track is not None:
                        for trackpoint in track.findall(f"{ns}Trackpoint"):
                            try:
                                point = TCXParser._parse_trackpoint(trackpoint, ns)
                                result.points.append(point)
                            except Exception as e:
                                error_msg = f"Failed to parse TCX trackpoint: {e}"
                                result.parse_errors.append(error_msg)
                                logger.debug(error_msg)
            
            if len(result.points) >= 2:
                GPXParser._calculate_statistics(result)
            
            logger.info(f" Parsed TCX: {len(result.points)} points, {len(result.parse_errors)} errors")
            
        except ET.ParseError as e:
            error_msg = f"XML parsing error: {e}"
            result.parse_errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing TCX: {e}"
            result.parse_errors.append(error_msg)
            logger.error(error_msg)
        
        return result
    
    @staticmethod
    def _parse_trackpoint(trackpoint: ET.Element, ns: str) -> TrackPoint:
        time_elem = trackpoint.find(f"{ns}Time")
        timestamp = GPXParser._parse_time(time_elem.text) if time_elem is not None and time_elem.text else None
        
        position = trackpoint.find(f"{ns}Position")
        if position is None:
            raise ValueError("Trackpoint missing Position element")
        
        lat_elem = position.find(f"{ns}LatitudeDegrees")
        lon_elem = position.find(f"{ns}LongitudeDegrees")
        
        if lat_elem is None or lon_elem is None:
            raise ValueError("Position missing lat/lon elements")
        
        lat = float(lat_elem.text)
        lon = float(lon_elem.text)
        
        alt_elem = trackpoint.find(f"{ns}AltitudeMeters")
        elevation = float(alt_elem.text) if alt_elem is not None and alt_elem.text else None
        
        return TrackPoint(
            latitude=lat,
            longitude=lon,
            elevation=elevation,
            timestamp=timestamp
        )



class GeoJSONParser:
    
    @staticmethod
    def parse(file_path: str) -> ParsedTrack:
        result = ParsedTrack(format="geojson")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features = []
            if data.get('type') == 'FeatureCollection':
                features = data.get('features', [])
            elif data.get('type') == 'Feature':
                features = [data]
            else:
                result.parse_errors.append(f"Unknown GeoJSON type: {data.get('type')}")
                return result
            
            if features:
                props = features[0].get('properties', {})
                result.metadata.name = props.get('name')
                result.metadata.description = props.get('description')
                result.metadata.activity_type = props.get('activity_type')
            
            for feature in features:
                geometry = feature.get('geometry', {})
                geom_type = geometry.get('type')
                coords = geometry.get('coordinates', [])
                
                if geom_type == 'LineString':
                    GeoJSONParser._parse_linestring(coords, result)
                elif geom_type == 'MultiLineString':
                    for line in coords:
                        GeoJSONParser._parse_linestring(line, result)
                else:
                    result.parse_errors.append(f"Unsupported geometry type: {geom_type}")
            
            if len(result.points) >= 2:
                GPXParser._calculate_statistics(result)
            
            logger.info(f" Parsed GeoJSON: {len(result.points)} points, {len(result.parse_errors)} errors")
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            result.parse_errors.append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing GeoJSON: {e}"
            result.parse_errors.append(error_msg)
            logger.error(error_msg)
        
        return result
    
    @staticmethod
    def _parse_linestring(coords: List, result: ParsedTrack):
        for coord in coords:
            try:
                lon = float(coord[0])
                lat = float(coord[1])
                elevation = float(coord[2]) if len(coord) > 2 else None
                
                point = TrackPoint(
                    latitude=lat,
                    longitude=lon,
                    elevation=elevation
                )
                result.points.append(point)
            except Exception as e:
                result.parse_errors.append(f"Failed to parse coordinate: {e}")



class UniversalTrackParser:
    
    @staticmethod
    def parse(file_path: str, bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> ParsedTrack:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            result = ParsedTrack()
            result.parse_errors.append(f"File not found: {file_path}")
            return result
        
        extension = file_path_obj.suffix.lower()
        
        if extension == '.gpx':
            result = GPXParser.parse(file_path)
        elif extension == '.tcx':
            result = TCXParser.parse(file_path)
        elif extension in ['.json', '.geojson']:
            result = GeoJSONParser.parse(file_path)
        else:
            result = ParsedTrack()
            result.parse_errors.append(f"Unsupported file format: {extension}")
            return result
        
        if bounds and result.points:
            result = UniversalTrackParser._validate_bounds(result, bounds)
        
        if result.points:
            result.points = UniversalTrackParser._remove_duplicates(result.points)
        
        return result
    
    @staticmethod
    def _validate_bounds(track: ParsedTrack, bounds: Dict[str, Tuple[float, float]]) -> ParsedTrack:
        original_count = len(track.points)
        valid_points = []
        
        lat_min, lat_max = bounds.get('latitude', (-90, 90))
        lon_min, lon_max = bounds.get('longitude', (-180, 180))
        ele_min, ele_max = bounds.get('elevation', (-500, 9000))
        
        for point in track.points:
            valid = True
            
            if not (lat_min <= point.latitude <= lat_max):
                valid = False
            
            if not (lon_min <= point.longitude <= lon_max):
                valid = False
            
            if point.elevation is not None and not (ele_min <= point.elevation <= ele_max):
                valid = False
            
            if valid:
                valid_points.append(point)
        
        removed_count = original_count - len(valid_points)
        if removed_count > 0:
            warning = f"Removed {removed_count} points outside bounds (lat: {lat_min}-{lat_max}, lon: {lon_min}-{lon_max})"
            track.parse_errors.append(warning)
            logger.warning(warning)
        
        track.points = valid_points
        return track
    
    @staticmethod
    def _remove_duplicates(points: List[TrackPoint], tolerance: float = 1e-6) -> List[TrackPoint]:
        if len(points) < 2:
            return points
        
        unique_points = [points[0]]
        
        for i in range(1, len(points)):
            prev = unique_points[-1]
            curr = points[i]
            
            lat_diff = abs(curr.latitude - prev.latitude)
            lon_diff = abs(curr.longitude - prev.longitude)
            
            if lat_diff > tolerance or lon_diff > tolerance:
                unique_points.append(curr)
        
        removed = len(points) - len(unique_points)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate points")
        
        return unique_points



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    bounds = {
        'latitude': (52.0, 60.0),
        'longitude': (109.0, 134.0),
        'elevation': (-100, 6000)
    }
    
    parser = UniversalTrackParser()
    
    track = parser.parse("example_track.gpx", bounds=bounds)
    
    if track.is_valid():
        print(f" Successfully parsed {track.format.upper()} track")
        print(f"   Points: {len(track.points)}")
        print(f"   Name: {track.metadata.name}")
        print(f"   Distance: {track.metadata.total_distance_m}m")
        print(f"   Elevation gain: {track.metadata.elevation_gain_m}m")
    else:
        print(f" Parsing failed with {len(track.parse_errors)} errors:")
        for error in track.parse_errors:
            print(f"   - {error}")

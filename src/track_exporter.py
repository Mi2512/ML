
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json

logger = logging.getLogger(__name__)



@dataclass
class ExportGeoPoint:
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class ExportTrackMetadata:
    name: Optional[str] = None
    description: Optional[str] = None
    activity_type: Optional[str] = None
    creator: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None



class GPXExporter:
    
    NS_GPX = "http://www.topografix.com/GPX/1/1"
    SCHEMA_LOCATION = "http://www.topografix.com/GPX/1/1/gpx.xsd"
    
    @staticmethod
    def create_gpx_root(metadata: ExportTrackMetadata) -> ET.Element:
        root = ET.Element('gpx')
        root.set('version', '1.1')
        root.set('creator', metadata.creator or 'Track Analysis System v1.5')
        root.set('xmlns', GPXExporter.NS_GPX)
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:schemaLocation', f'{GPXExporter.NS_GPX} {GPXExporter.SCHEMA_LOCATION}')
        
        return root
    
    @staticmethod
    def _datetime_to_iso8601(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')
    
    @staticmethod
    def export_track_to_gpx(
        track_id: str,
        points: List[ExportGeoPoint],
        metadata: ExportTrackMetadata,
        output_path: str
    ) -> bool:
        try:
            if not points:
                logger.error(f"No points to export for track {track_id}")
                return False
            
            for point in points:
                if not (-90 <= point.latitude <= 90):
                    logger.error(f"Invalid latitude: {point.latitude}")
                    return False
                if not (-180 <= point.longitude <= 180):
                    logger.error(f"Неверная долгота: {point.longitude}")
                    return False
            
            root = GPXExporter.create_gpx_root(metadata)
            
            metadata_elem = ET.SubElement(root, 'metadata')
            
            if metadata.name:
                name_elem = ET.SubElement(metadata_elem, 'name')
                name_elem.text = str(metadata.name)
            
            if metadata.description:
                desc_elem = ET.SubElement(metadata_elem, 'desc')
                desc_elem.text = str(metadata.description)
            
            if metadata.start_time:
                time_elem = ET.SubElement(metadata_elem, 'time')
                time_elem.text = GPXExporter._datetime_to_iso8601(metadata.start_time)
            
            trk = ET.SubElement(root, 'trk')
            
            if metadata.name:
                name_elem = ET.SubElement(trk, 'name')
                name_elem.text = str(metadata.name)
            
            trkseg = ET.SubElement(trk, 'trkseg')
            
            for point in points:
                trkpt = ET.SubElement(trkseg, 'trkpt')
                trkpt.set('lat', f'{point.latitude:.8f}')
                trkpt.set('lon', f'{point.longitude:.8f}')
                
                if point.elevation is not None:
                    ele = ET.SubElement(trkpt, 'ele')
                    ele.text = f'{point.elevation:.2f}'
                
                if point.timestamp:
                    time = ET.SubElement(trkpt, 'time')
                    time.text = GPXExporter._datetime_to_iso8601(point.timestamp)
            
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
            xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            logger.info(f"Экспорт GPX: {output_path} ({len(points)} точек)")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка GPX {track_id}: {e}")
            return False



class TCXExporter:
    
    NS_TCX = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    
    @staticmethod
    def create_tcx_root(creator: str = 'Track Analysis System v1.5') -> ET.Element:
        root = ET.Element('TrainingCenterDatabase')
        root.set('xmlns', TCXExporter.NS_TCX)
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        
        return root
    
    @staticmethod
    def _datetime_to_iso8601(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace('+00:00', 'Z')
    
    @staticmethod
    def _get_activity_type(activity: Optional[str]) -> str:
        activity_map = {
            'run': 'Running',
            'running': 'Running',
            'hiking': 'Walking',
            'hike': 'Walking',
            'walk': 'Walking',
            'walking': 'Walking',
            'cycling': 'Biking',
            'bike': 'Biking',
            'bicycle': 'Biking',
            'biking': 'Biking',
            'swimming': 'Swimming',
            'swim': 'Swimming',
            'skiing': 'Skiing',
            'ski': 'Skiing',
            'mountaineering': 'Walking',
            'climbing': 'Walking',
        }
        
        if activity is None:
            return 'Other'
        
        activity_lower = activity.lower().strip()
        return activity_map.get(activity_lower, 'Other')
    
    @staticmethod
    def export_track_to_tcx(
        track_id: str,
        points: List[ExportGeoPoint],
        metadata: ExportTrackMetadata,
        output_path: str
    ) -> bool:
        try:
            if not points:
                logger.error(f"No points to export for track {track_id}")
                return False
            
            for point in points:
                if not (-90 <= point.latitude <= 90):
                    logger.error(f"Invalid latitude: {point.latitude}")
                    return False
                if not (-180 <= point.longitude <= 180):
                    logger.error(f"Invalid longitude: {point.longitude}")
                    return False
            
            root = TCXExporter.create_tcx_root()
            
            activities = ET.SubElement(root, 'Activities')
            
            activity_type = TCXExporter._get_activity_type(metadata.activity_type)
            
            activity = ET.SubElement(activities, 'Activity')
            activity.set('Sport', activity_type)
            
            id_elem = ET.SubElement(activity, 'Id')
            if metadata.start_time:
                id_elem.text = TCXExporter._datetime_to_iso8601(metadata.start_time)
            else:
                id_elem.text = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            lap = ET.SubElement(activity, 'Lap')
            if metadata.start_time:
                lap.set('StartTime', TCXExporter._datetime_to_iso8601(metadata.start_time))
            
            total_time_seconds = 0
            if metadata.start_time and metadata.end_time:
                delta = metadata.end_time - metadata.start_time
                total_time_seconds = delta.total_seconds()
            
            time_elem = ET.SubElement(lap, 'TotalTimeSeconds')
            time_elem.text = f'{total_time_seconds:.1f}'
            
            distance_elem = ET.SubElement(lap, 'DistanceMeters')
            distance_elem.text = '0'
            
            intensity_elem = ET.SubElement(lap, 'Intensity')
            intensity_elem.text = 'Active'
            
            cadence_elem = ET.SubElement(lap, 'Cadence')
            cadence_elem.text = '0'
            
            track = ET.SubElement(lap, 'Track')
            
            for point in points:
                tp = ET.SubElement(track, 'Trackpoint')
                
                if point.timestamp:
                    time_tp = ET.SubElement(tp, 'Time')
                    time_tp.text = TCXExporter._datetime_to_iso8601(point.timestamp)
                
                position = ET.SubElement(tp, 'Position')
                
                lat_elem = ET.SubElement(position, 'LatitudeDegrees')
                lat_elem.text = f'{point.latitude:.8f}'
                
                lon_elem = ET.SubElement(position, 'LongitudeDegrees')
                lon_elem.text = f'{point.longitude:.8f}'
                
                if point.elevation is not None:
                    alt_elem = ET.SubElement(tp, 'AltitudeMeters')
                    alt_elem.text = f'{point.elevation:.2f}'
            
            extensions = ET.SubElement(activity, 'Extensions')
            
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
            xml_str = '\n'.join([line for line in xml_str.split('\n') if line.strip()])
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            logger.info(f"Экспорт TCX: {output_path} ({len(points)} точек)")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка TCX {track_id}: {e}")
            return False



class TrackExporter:
    
    @staticmethod
    def export_track(
        track_id: str,
        points: List[ExportGeoPoint],
        metadata: ExportTrackMetadata,
        output_dir: str = 'data/exports',
        formats: List[str] = None
    ) -> Dict[str, bool]:
        if formats is None:
            formats = ['gpx', 'tcx']
        
        results = {}
        output_dir = Path(output_dir)
        
        for fmt in formats:
            if fmt.lower() == 'gpx':
                output_path = output_dir / f'{track_id}.gpx'
                results['gpx'] = GPXExporter.export_track_to_gpx(
                    track_id, points, metadata, str(output_path)
                )
            
            elif fmt.lower() == 'tcx':
                output_path = output_dir / f'{track_id}.tcx'
                results['tcx'] = TCXExporter.export_track_to_tcx(
                    track_id, points, metadata, str(output_path)
                )
        
        return results
    
    @staticmethod
    def batch_export_tracks(
        tracks: List[Tuple[str, List[ExportGeoPoint], ExportTrackMetadata]],
        output_dir: str = 'data/exports',
        formats: List[str] = None
    ) -> Dict[str, Dict[str, bool]]:
        if formats is None:
            formats = ['gpx', 'tcx']
        
        results = {}
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Экспорт {len(tracks)} треков")
        logger.info(f"{'='*70}")
        
        for i, (track_id, points, metadata) in enumerate(tracks, 1):
            logger.info(f"\n{i}/{len(tracks)}: {track_id}")
            results[track_id] = TrackExporter.export_track(
                track_id, points, metadata, output_dir, formats
            )
            
            successful_formats = [fmt for fmt, success in results[track_id].items() if success]
            if successful_formats:
                logger.info(f"  Готово: {', '.join(successful_formats)}")
            else:
                logger.warning(f"  Не удалось")
        
        return results



def export_track_from_db(
    db,
    track_id: str,
    output_dir: str = 'data/exports',
    formats: List[str] = None
) -> Dict[str, bool]:
    from augmentation_integration import track_from_db_to_geopoints
    
    try:
        geo_points = track_from_db_to_geopoints(db, track_id)
    except Exception as e:
        logger.error(f"Не удалось загрузить {track_id}: {e}")
        return {fmt: False for fmt in (formats or ['gpx', 'tcx'])}
    
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
        creator="Track Analysis System v1.5"
    )
    
    return TrackExporter.export_track(
        track_id, export_points, metadata, output_dir, formats
    )


def batch_export_from_db(
    db,
    track_ids: List[str],
    output_dir: str = 'data/exports',
    formats: List[str] = None
) -> Dict[str, Dict[str, bool]]:
    from augmentation_integration import track_from_db_to_geopoints
    
    tracks_to_export = []
    
    for track_id in track_ids:
        try:
            geo_points = track_from_db_to_geopoints(db, track_id)
            
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
                creator="Track Analysis System v1.5"
            )
            
            tracks_to_export.append((track_id, export_points, metadata))
            
        except Exception as e:
            logger.warning(f"Ошибка загрузки {track_id}: {e}")
    
    return TrackExporter.batch_export_tracks(
        tracks_to_export, output_dir, formats
    )



if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("Track Exporter - Test")
    print("="*70)
    
    test_points = [
        ExportGeoPoint(latitude=56.0, longitude=125.0, elevation=100.0,
                      timestamp=datetime(2026, 1, 15, 10, 0, 0)),
        ExportGeoPoint(latitude=56.001, longitude=125.001, elevation=110.0,
                      timestamp=datetime(2026, 1, 15, 10, 1, 0)),
        ExportGeoPoint(latitude=56.002, longitude=125.002, elevation=120.0,
                      timestamp=datetime(2026, 1, 15, 10, 2, 0)),
        ExportGeoPoint(latitude=56.003, longitude=125.003, elevation=115.0,
                      timestamp=datetime(2026, 1, 15, 10, 3, 0)),
        ExportGeoPoint(latitude=56.004, longitude=125.004, elevation=105.0,
                      timestamp=datetime(2026, 1, 15, 10, 4, 0)),
    ]
    
    test_metadata = ExportTrackMetadata(
        name="Test Track",
        description="Test hiking track in Russia Far East",
        activity_type="hiking",
        creator="Track Analysis System v1.5",
        start_time=datetime(2026, 1, 15, 10, 0, 0),
        end_time=datetime(2026, 1, 15, 10, 4, 0)
    )
    
    print("\nTest 1: GPX Export")
    print("-" * 70)
    success = GPXExporter.export_track_to_gpx(
        "TEST_TRACK_001",
        test_points,
        test_metadata,
        "test_export/TEST_TRACK_001.gpx"
    )
    print(f"GPX Export: {' Success' if success else ' Failed'}")
    
    print("\nTest 2: TCX Export")
    print("-" * 70)
    success = TCXExporter.export_track_to_tcx(
        "TEST_TRACK_001",
        test_points,
        test_metadata,
        "test_export/TEST_TRACK_001.tcx"
    )
    print(f"TCX Export: {' Success' if success else ' Failed'}")
    
    print("\nTest 3: Batch Export")
    print("-" * 70)
    
    tracks = [
        ("TRACK_A", test_points, test_metadata),
        ("TRACK_B", test_points, ExportTrackMetadata(
            name="Track B",
            description="Test cycling track",
            activity_type="cycling",
            start_time=datetime(2026, 1, 16, 10, 0, 0),
            end_time=datetime(2026, 1, 16, 10, 4, 0)
        )),
    ]
    
    results = TrackExporter.batch_export_tracks(
        tracks,
        output_dir="test_export",
        formats=["gpx", "tcx"]
    )
    
    print("\nBatch Export Results:")
    for track_id, format_results in results.items():
        print(f"  {track_id}: {format_results}")
    
    print("\n" + "="*70)
    print(" Export tests Готово!")
    print("="*70)

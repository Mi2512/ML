
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import logging
import hashlib
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, 
    Boolean, JSON, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
import pandas as pd

logger = logging.getLogger(__name__)

Base = declarative_base()


class RouteTrack(Base):
    __tablename__ = 'route_tracks'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String(50), unique=True, nullable=False, index=True)
    region = Column(String(100))
    date_start = Column(DateTime)
    date_end = Column(DateTime)
    point_count = Column(Integer)
    distance_km = Column(Float)
    elevation_gain_m = Column(Float)
    elevation_loss_m = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    
    points = relationship('RoutePoint', back_populates='track', cascade='all, delete-orphan')
    anomalies = relationship('GPSAnomaly', back_populates='track', cascade='all, delete-orphan')
    corrections = relationship('Correction', back_populates='track', cascade='all, delete-orphan')
    history_records = relationship('VersionHistory', back_populates='track', cascade='all, delete-orphan')
    external_sources = relationship('ExternalTrackSource', back_populates='track', foreign_keys='ExternalTrackSource.track_id_fk')
    original_augmentations = relationship('TrackAugmentation', back_populates='original_track', foreign_keys='TrackAugmentation.original_track_id')
    augmented_from = relationship('TrackAugmentation', back_populates='augmented_track', foreign_keys='TrackAugmentation.augmented_track_id')
    
    __table_args__ = (
        Index('idx_track_id', 'track_id'),
        Index('idx_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<RouteTrack(track_id={self.track_id}, points={self.point_count})>"


class RoutePoint(Base):
    __tablename__ = 'route_points'
    
    id = Column(Integer, primary_key=True)
    track_id_fk = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    point_index = Column(Integer, nullable=False)
    date = Column(DateTime)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude = Column(Float)
    frequency_steps = Column(Float)
    temperature = Column(Float)
    terrain_type = Column(String(50))
    surrounding_objects = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    version = Column(Integer, default=1)
    is_corrected = Column(Boolean, default=False)
    
    track = relationship('RouteTrack', back_populates='points')
    
    __table_args__ = (
        UniqueConstraint('track_id_fk', 'point_index', name='uq_track_point'),
        Index('idx_track_point', 'track_id_fk', 'point_index'),
        Index('idx_coordinates', 'latitude', 'longitude'),
    )
    
    def __repr__(self):
        return f"<RoutePoint(track={self.track_id_fk}, idx={self.point_index}, lat={self.latitude})>"


class GPSAnomaly(Base):
    __tablename__ = 'gps_anomalies'
    
    id = Column(Integer, primary_key=True)
    track_id_fk = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    point_index = Column(Integer)
    anomaly_type = Column(String(50), nullable=False)
    description = Column(String(500))
    severity = Column(String(20), default='medium')
    original_lat = Column(Float)
    original_lon = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow)
    is_corrected = Column(Boolean, default=False)
    
    track = relationship('RouteTrack', back_populates='anomalies')
    
    __table_args__ = (
        Index('idx_anomaly_track', 'track_id_fk'),
        Index('idx_anomaly_type', 'anomaly_type'),
    )
    
    def __repr__(self):
        return f"<GPSAnomaly(track={self.track_id_fk}, type={self.anomaly_type})>"


class Correction(Base):
    __tablename__ = 'corrections'
    
    id = Column(Integer, primary_key=True)
    track_id_fk = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    point_index = Column(Integer)
    correction_type = Column(String(50), nullable=False)
    method = Column(String(100))
    original_value = Column(JSON)
    corrected_value = Column(JSON)
    confidence = Column(Float)
    applied_at = Column(DateTime, default=datetime.utcnow)
    applied_by = Column(String(100), default='auto')
    
    track = relationship('RouteTrack', back_populates='corrections')
    
    __table_args__ = (
        Index('idx_correction_track', 'track_id_fk'),
        Index('idx_correction_type', 'correction_type'),
    )
    
    def __repr__(self):
        return f"<Correction(track={self.track_id_fk}, type={self.correction_type})>"


class VersionHistory(Base):
    __tablename__ = 'version_history'
    
    id = Column(Integer, primary_key=True)
    track_id_fk = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    change_type = Column(String(50), nullable=False)
    changed_fields = Column(JSON)
    change_description = Column(String(500))
    change_hash = Column(String(64))
    changed_by = Column(String(100), default='system')
    changed_at = Column(DateTime, default=datetime.utcnow)
    
    track = relationship('RouteTrack', back_populates='history_records')
    
    __table_args__ = (
        Index('idx_history_track', 'track_id_fk'),
        Index('idx_history_version', 'track_id_fk', 'version_number'),
        Index('idx_history_time', 'changed_at'),
    )
    
    def __repr__(self):
        return f"<VersionHistory(track={self.track_id_fk}, v{self.version_number})>"


class TrackMap(Base):
    __tablename__ = 'track_maps'
    
    id = Column(Integer, primary_key=True)
    track_id_fk = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    map_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    map_source = Column(String(100), default='OpenStreetMap')
    center_latitude = Column(Float)
    center_longitude = Column(Float)
    zoom_level = Column(Integer)
    has_overlay = Column(Boolean, default=True)
    has_geodesic = Column(Boolean, default=True)
    overlay_points_count = Column(Integer)
    projection_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    map_metadata = Column(JSON)
    
    track = relationship('RouteTrack', foreign_keys=[track_id_fk])
    
    __table_args__ = (
        Index('idx_map_track', 'track_id_fk'),
        Index('idx_map_type', 'map_type'),
        Index('idx_map_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TrackMap(track={self.track_id_fk}, type={self.map_type}, path={self.file_path})>"


class ExternalTrackSource(Base):
    __tablename__ = 'external_track_sources'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50), nullable=False, index=True)
    source_id = Column(String(100), nullable=False)
    import_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    original_url = Column(String(500))
    platform_metadata = Column(JSON)
    file_format = Column(String(20))
    file_path = Column(String(500))
    track_id_fk = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=True, index=True)
    is_processed = Column(Boolean, default=False)
    processing_errors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    track = relationship('RouteTrack', foreign_keys=[track_id_fk])
    
    __table_args__ = (
        UniqueConstraint('platform', 'source_id', name='uq_platform_source'),
        Index('idx_external_platform', 'platform'),
        Index('idx_external_import_date', 'import_date'),
        Index('idx_external_processed', 'is_processed'),
    )
    
    def __repr__(self):
        return f"<ExternalTrackSource(platform={self.platform}, id={self.source_id})>"


class TrackAugmentation(Base):
    __tablename__ = 'track_augmentations'
    
    id = Column(Integer, primary_key=True)
    original_track_id = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    augmented_track_id = Column(String(50), ForeignKey('route_tracks.track_id'), nullable=False, index=True)
    augmentation_type = Column(String(50), nullable=False, index=True)
    parameters = Column(JSON, nullable=False)
    transformation_matrix = Column(JSON)
    is_reversible = Column(Boolean, default=True)
    validation_status = Column(String(20), default='pending')
    validation_errors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100), default='system')
    
    original_track = relationship('RouteTrack', foreign_keys=[original_track_id])
    augmented_track = relationship('RouteTrack', foreign_keys=[augmented_track_id])
    
    __table_args__ = (
        UniqueConstraint('original_track_id', 'augmented_track_id', name='uq_original_augmented'),
        Index('idx_aug_original', 'original_track_id'),
        Index('idx_aug_type', 'augmentation_type'),
        Index('idx_aug_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TrackAugmentation(original={self.original_track_id}, aug={self.augmented_track_id}, type={self.augmentation_type})>"


class DatabaseManager:
    
    def __init__(self, db_path: str = 'data/db/routes.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        db_url = f'sqlite:///{self.db_path}'
        self.engine = create_engine(db_url, echo=False)
        
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_track(self, track_id: str, region: str = None, 
                 date_start: datetime = None, date_end: datetime = None) -> str:
        session = self.Session()
        
        try:
            existing = session.query(RouteTrack).filter_by(track_id=track_id).first()
            if existing:
                logger.warning(f"Track {track_id} already exists, returning existing")
                return existing.track_id
            
            track = RouteTrack(
                track_id=track_id,
                region=region,
                date_start=date_start,
                date_end=date_end,
                version=1
            )
            
            session.add(track)
            session.commit()
            
            session.refresh(track)
            result_track_id = track.track_id
            
            logger.info(f"Added track {result_track_id}")
            
            self._log_version_history(session, track_id, 1, 'create', None, f"Track created")
            
            return result_track_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding track: {e}")
            raise
        finally:
            session.close()
    
    def add_points(self, track_id: str, points_data: List[Dict]) -> int:
        session = self.Session()
        
        try:
            track = session.query(RouteTrack).filter_by(track_id=track_id).first()
            if not track:
                raise ValueError(f"Track {track_id} not found")
            
            existing_indices = set()
            for existing_point in session.query(RoutePoint.point_index).filter_by(track_id_fk=track_id).all():
                existing_indices.add(existing_point[0])
            
            added_count = 0
            skipped_count = 0
            
            for point_data in points_data:
                point_index = point_data.get('point_index')
                
                if point_index in existing_indices:
                    skipped_count += 1
                    continue
                
                point = RoutePoint(
                    track_id_fk=track_id,
                    point_index=point_index,
                    date=point_data.get('date'),
                    latitude=point_data.get('latitude'),
                    longitude=point_data.get('longitude'),
                    altitude=point_data.get('altitude'),
                    frequency_steps=point_data.get('frequency_steps'),
                    temperature=point_data.get('temperature'),
                    terrain_type=point_data.get('terrain_type'),
                    surrounding_objects=point_data.get('surrounding_objects'),
                    version=1
                )
                session.add(point)
                added_count += 1
            
            if added_count > 0:
                session.commit()
                logger.info(f"Added {added_count} points to track {track_id}" + 
                           (f" (skipped {skipped_count} existing points)" if skipped_count > 0 else ""))
                
                track.point_count = session.query(RoutePoint).filter_by(track_id_fk=track_id).count()
                track.updated_at = datetime.utcnow()
                session.commit()
            elif skipped_count > 0:
                logger.info(f"Track {track_id}: all {skipped_count} points already exist, skipping")
            
            return added_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding points: {e}")
            raise
        finally:
            session.close()
    
    def record_anomaly(self, track_id: str, point_index: int,
                      anomaly_type: str, description: str = None,
                      severity: str = 'medium',
                      original_lat: float = None,
                      original_lon: float = None) -> GPSAnomaly:
        session = self.Session()
        
        try:
            anomaly = GPSAnomaly(
                track_id_fk=track_id,
                point_index=point_index,
                anomaly_type=anomaly_type,
                description=description,
                severity=severity,
                original_lat=original_lat,
                original_lon=original_lon,
                is_corrected=False
            )
            
            session.add(anomaly)
            session.commit()
            logger.info(f"Recorded anomaly in track {track_id} at point {point_index}")
            
            return anomaly
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording anomaly: {e}")
            raise
        finally:
            session.close()
    
    def apply_correction(self, track_id: str, point_index: int,
                        correction_type: str, method: str,
                        original_value: Dict, corrected_value: Dict,
                        confidence: float = 0.8) -> Correction:
        session = self.Session()
        
        try:
            correction = Correction(
                track_id_fk=track_id,
                point_index=point_index,
                correction_type=correction_type,
                method=method,
                original_value=original_value,
                corrected_value=corrected_value,
                confidence=confidence,
                applied_by='auto'
            )
            
            session.add(correction)
            
            point = session.query(RoutePoint).filter_by(
                track_id_fk=track_id, point_index=point_index
            ).first()
            if point:
                point.is_corrected = True
                point.version += 1
            
            anomaly = session.query(GPSAnomaly).filter_by(
                track_id_fk=track_id, point_index=point_index
            ).first()
            if anomaly:
                anomaly.is_corrected = True
            
            session.commit()
            logger.info(f"Applied {correction_type} correction to {track_id}")
            
            self._increment_track_version(session, track_id)
            
            return correction
        except Exception as e:
            session.rollback()
            logger.error(f"Error applying correction: {e}")
            raise
        finally:
            session.close()
    
    def get_track_history(self, track_id: str) -> List[VersionHistory]:
        session = self.Session()
        
        try:
            history = session.query(VersionHistory).filter_by(
                track_id_fk=track_id
            ).order_by(VersionHistory.version_number).all()
            
            return history
        finally:
            session.close()
    
    def _log_version_history(self, session: Session, track_id: str, version: int,
                            change_type: str, changed_fields: Dict,
                            description: str) -> VersionHistory:
        change_str = json.dumps(changed_fields or {}, sort_keys=True)
        change_hash = hashlib.sha256(change_str.encode()).hexdigest()
        
        history = VersionHistory(
            track_id_fk=track_id,
            version_number=version,
            change_type=change_type,
            changed_fields=changed_fields,
            change_description=description,
            change_hash=change_hash,
            changed_by='system'
        )
        
        session.add(history)
        session.commit()
        
        return history
    
    def _increment_track_version(self, session: Session, track_id: str) -> None:
        track = session.query(RouteTrack).filter_by(track_id=track_id).first()
        if track:
            track.version += 1
            track.updated_at = datetime.utcnow()
            session.commit()
    
    def export_track_to_dataframe(self, track_id: str) -> pd.DataFrame:
        session = self.Session()
        
        try:
            points = session.query(RoutePoint).filter_by(
                track_id_fk=track_id
            ).order_by(RoutePoint.point_index).all()
            
            data = []
            for point in points:
                data.append({
                    'point_index': point.point_index,
                    'date': point.date,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'altitude': point.altitude,
                    'frequency_steps': point.frequency_steps,
                    'temperature': point.temperature,
                    'terrain_type': point.terrain_type,
                    'surrounding_objects': point.surrounding_objects,
                    'is_corrected': point.is_corrected,
                    'version': point.version
                })
            
            df = pd.DataFrame(data)
            return df
        finally:
            session.close()
    
    def get_database_statistics(self) -> Dict:
        session = self.Session()
        
        try:
            stats = {
                'total_tracks': session.query(RouteTrack).count(),
                'total_points': session.query(RoutePoint).count(),
                'total_anomalies': session.query(GPSAnomaly).count(),
                'corrected_anomalies': session.query(GPSAnomaly).filter_by(is_corrected=True).count(),
                'total_corrections': session.query(Correction).count(),
                'total_version_records': session.query(VersionHistory).count(),
                'total_maps': session.query(TrackMap).count(),
                'total_external_sources': session.query(ExternalTrackSource).count(),
                'total_augmentations': session.query(TrackAugmentation).count()
            }
            
            return stats
        finally:
            session.close()
    
    def get_all_points_as_dataframe(self) -> Optional[pd.DataFrame]:
        session = self.Session()
        
        try:
            points = session.query(RoutePoint).order_by(
                RoutePoint.track_id_fk, RoutePoint.point_index
            ).all()
            
            if not points:
                logger.warning("No points found in database")
                return None
            
            data = []
            for point in points:
                point_dict = {
                    'track_id': point.track_id_fk,
                    'point_index': point.point_index,
                    'date': point.date,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'altitude': point.altitude,
                    'frequency_steps': point.frequency_steps,
                    'temperature': point.temperature,
                    'terrain_type': point.terrain_type,
                    'surrounding_objects': point.surrounding_objects,
                    'is_corrected': point.is_corrected,
                    'version': point.version
                }
                data.append(point_dict)
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} points from database into DataFrame")
            return df
        except Exception as e:
            logger.error(f"Error loading points from database: {e}")
            return None
        finally:
            session.close()
    
    def save_map_to_db(self, track_id: str, map_type: str, file_path: str, 
                       map_source: str = 'OpenStreetMap', center_lat: float = None,
                       center_lon: float = None, zoom_level: int = None,
                       has_overlay: bool = True, has_geodesic: bool = True,
                       overlay_points_count: int = None, projection_info: Dict = None,
                       metadata: Dict = None) -> TrackMap:
        session = self.Session()
        
        try:
            track = session.query(RouteTrack).filter_by(track_id=track_id).first()
            if not track:
                raise ValueError(f"Track {track_id} not found in database")
            
            track_map = TrackMap(
                track_id_fk=track_id,
                map_type=map_type,
                file_path=str(file_path),
                map_source=map_source,
                center_latitude=center_lat,
                center_longitude=center_lon,
                zoom_level=zoom_level,
                has_overlay=has_overlay,
                has_geodesic=has_geodesic,
                overlay_points_count=overlay_points_count,
                projection_info=projection_info or {},
                map_metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            session.add(track_map)
            session.commit()
            
            logger.info(f"Saved {map_type} map for track {track_id} to database: {file_path}")
            
            return track_map
        finally:
            session.close()
    
    def get_track_maps(self, track_id: str) -> List[Dict]:
        session = self.Session()
        
        try:
            maps = session.query(TrackMap).filter_by(
                track_id_fk=track_id,
                is_active=True
            ).order_by(TrackMap.created_at.desc()).all()
            
            result = []
            for m in maps:
                result.append({
                    'id': m.id,
                    'track_id': m.track_id_fk,
                    'map_type': m.map_type,
                    'file_path': m.file_path,
                    'map_source': m.map_source,
                    'center': (m.center_latitude, m.center_longitude) if m.center_latitude else None,
                    'zoom_level': m.zoom_level,
                    'has_overlay': m.has_overlay,
                    'has_geodesic': m.has_geodesic,
                    'overlay_points': m.overlay_points_count,
                    'projection_info': m.projection_info,
                    'created_at': m.created_at.isoformat() if m.created_at else None,
                    'version': m.version,
                    'metadata': m.map_metadata
                })
            
            return result
        finally:
            session.close()
    
    def add_external_track_source(self, platform: str, source_id: str, 
                                   platform_metadata: Dict = None, original_url: str = None,
                                   file_format: str = None, file_path: str = None,
                                   track_id: str = None) -> ExternalTrackSource:
        session = self.Session()
        
        try:
            existing = session.query(ExternalTrackSource).filter_by(
                platform=platform,
                source_id=source_id
            ).first()
            
            if existing:
                logger.warning(f"External track {platform}/{source_id} already exists (ID: {existing.id})")
                return existing
            
            external_source = ExternalTrackSource(
                platform=platform.lower(),
                source_id=str(source_id),
                import_date=datetime.utcnow(),
                original_url=original_url,
                platform_metadata=platform_metadata or {},
                file_format=file_format,
                file_path=file_path,
                track_id_fk=track_id,
                is_processed=bool(track_id)
            )
            
            session.add(external_source)
            session.commit()
            
            session.refresh(external_source)
            result_id = external_source.id
            result_platform = external_source.platform
            result_source_id = external_source.source_id
            
            logger.info(f"Added external track source: {result_platform}/{result_source_id} (ID: {result_id})")
            
            return external_source
        finally:
            session.close()
    
    def add_track_augmentation(self, original_track_id: str, augmented_track_id: str,
                               augmentation_type: str, parameters: Dict,
                               transformation_matrix: Dict = None,
                               is_reversible: bool = True) -> TrackAugmentation:
        session = self.Session()
        
        try:
            original = session.query(RouteTrack).filter_by(track_id=original_track_id).first()
            augmented = session.query(RouteTrack).filter_by(track_id=augmented_track_id).first()
            
            if not original:
                raise ValueError(f"Original track {original_track_id} not found")
            if not augmented:
                raise ValueError(f"Augmented track {augmented_track_id} not found")
            
            existing = session.query(TrackAugmentation).filter_by(
                original_track_id=original_track_id,
                augmented_track_id=augmented_track_id
            ).first()
            
            if existing:
                logger.warning(f"Augmentation {original_track_id} -> {augmented_track_id} already exists")
                return existing
            
            augmentation = TrackAugmentation(
                original_track_id=original_track_id,
                augmented_track_id=augmented_track_id,
                augmentation_type=augmentation_type,
                parameters=parameters,
                transformation_matrix=transformation_matrix or {},
                is_reversible=is_reversible,
                validation_status='pending',
                created_at=datetime.utcnow()
            )
            
            session.add(augmentation)
            session.commit()
            
            session.refresh(augmentation)
            result_id = augmentation.id
            result_type = augmentation.augmentation_type
            
            logger.info(f"Added augmentation: {original_track_id} -> {augmented_track_id} ({result_type}, ID: {result_id})")
            
            return augmentation
        finally:
            session.close()
    
    def get_track_augmentations(self, track_id: str, as_original: bool = True) -> List[Dict]:
        session = self.Session()
        
        try:
            if as_original:
                augmentations = session.query(TrackAugmentation).filter_by(
                    original_track_id=track_id
                ).order_by(TrackAugmentation.created_at.desc()).all()
            else:
                augmentations = session.query(TrackAugmentation).filter_by(
                    augmented_track_id=track_id
                ).order_by(TrackAugmentation.created_at.desc()).all()
            
            result = []
            for aug in augmentations:
                result.append({
                    'id': aug.id,
                    'original_track_id': aug.original_track_id,
                    'augmented_track_id': aug.augmented_track_id,
                    'augmentation_type': aug.augmentation_type,
                    'parameters': aug.parameters,
                    'transformation_matrix': aug.transformation_matrix,
                    'is_reversible': aug.is_reversible,
                    'validation_status': aug.validation_status,
                    'validation_errors': aug.validation_errors,
                    'created_at': aug.created_at.isoformat() if aug.created_at else None,
                    'created_by': aug.created_by
                })
            
            return result
        finally:
            session.close()
    
    def get_external_tracks(self, platform: str = None, is_processed: bool = None) -> List[Dict]:
        session = self.Session()
        
        try:
            query = session.query(ExternalTrackSource)
            
            if platform:
                query = query.filter_by(platform=platform.lower())
            if is_processed is not None:
                query = query.filter_by(is_processed=is_processed)
            
            sources = query.order_by(ExternalTrackSource.import_date.desc()).all()
            
            result = []
            for source in sources:
                result.append({
                    'id': source.id,
                    'platform': source.platform,
                    'source_id': source.source_id,
                    'import_date': source.import_date.isoformat() if source.import_date else None,
                    'original_url': source.original_url,
                    'platform_metadata': source.platform_metadata,
                    'file_format': source.file_format,
                    'file_path': source.file_path,
                    'track_id': source.track_id_fk,
                    'is_processed': source.is_processed,
                    'processing_errors': source.processing_errors
                })
            
            return result
        finally:
            session.close()

    
    def update_map_metadata(self, map_id: int, metadata: Dict) -> TrackMap:
        session = self.Session()
        
        try:
            track_map = session.query(TrackMap).filter_by(id=map_id).first()
            if not track_map:
                raise ValueError(f"Map with ID {map_id} not found")
            
            track_map.map_metadata = {**(track_map.map_metadata or {}), **metadata}
            track_map.updated_at = datetime.utcnow()
            track_map.version += 1
            
            session.commit()
            
            logger.info(f"Updated metadata for map ID {map_id}")
            
            return track_map
        finally:
            session.close()


if __name__ == "__main__":
    db_manager = DatabaseManager('data/db/test.db')
    
    track = db_manager.add_track('TEST_TRACK_001', region='Test Region')
    
    points = [
        {'point_index': 0, 'latitude': 55.75, 'longitude': 37.61, 'altitude': 100},
        {'point_index': 1, 'latitude': 55.751, 'longitude': 37.611, 'altitude': 105},
    ]
    db_manager.add_points('TEST_TRACK_001', points)
    
    track_map = db_manager.save_map_to_db(
        track_id='TEST_TRACK_001',
        map_type='topographic',
        file_path='data/maps/test_track_001_topo.html',
        map_source='OpenStreetMap',
        center_lat=55.751,
        center_lon=37.611,
        zoom_level=12,
        has_overlay=True,
        has_geodesic=True,
        overlay_points_count=2,
        projection_info={'distortion_factor': 1.02}
    )
    
    stats = db_manager.get_database_statistics()
    print("Database stats:", stats)
    
    maps = db_manager.get_track_maps('TEST_TRACK_001')
    print("Track maps:", maps)

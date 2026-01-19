
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)



@dataclass
class FeatureStatistics:
    feature_name: str
    data_type: str
    
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    
    null_count: int = 0
    null_percentage: float = 0.0
    
    unique_values: int = 0
    most_common: Optional[str] = None
    most_common_count: int = 0
    
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SeasonalDistribution:
    season: str
    total_points: int
    percentage: float
    expected_percentage: float
    status: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RareCombination:
    combination_key: str
    combination_dict: Dict[str, str]
    current_points: int
    expected_points: int
    deficit: int
    coverage_percentage: float
    action: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SeasonalityAnalysisResult:
    total_points: int
    seasonal_distributions: List[SeasonalDistribution]
    gini_coefficient: float
    rare_combinations: List[RareCombination]
    summary_status: str
    
    def to_dict(self) -> Dict:
        return {
            'total_points': self.total_points,
            'seasonal_distributions': [s.to_dict() for s in self.seasonal_distributions],
            'gini_coefficient': round(self.gini_coefficient, 4),
            'rare_combinations_count': len(self.rare_combinations),
            'rare_combinations': [r.to_dict() for r in self.rare_combinations[:20]],
            'summary_status': self.summary_status,
        }



class FeatureStatisticsCalculator:
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame, 
                            numerical_features: set,
                            categorical_features: set) -> Dict[str, FeatureStatistics]:
        stats = {}
        
        for col in df.columns:
            if col in ['track_id', 'point_index', 'extraction_timestamp']:
                continue
            
            null_count = df[col].isna().sum()
            null_percentage = (null_count / len(df)) * 100
            
            if col in numerical_features:
                data = df[col].dropna()
                
                stats[col] = FeatureStatistics(
                    feature_name=col,
                    data_type='numerical',
                    mean=float(data.mean()) if len(data) > 0 else None,
                    std=float(data.std()) if len(data) > 0 else None,
                    min=float(data.min()) if len(data) > 0 else None,
                    q25=float(data.quantile(0.25)) if len(data) > 0 else None,
                    median=float(data.median()) if len(data) > 0 else None,
                    q75=float(data.quantile(0.75)) if len(data) > 0 else None,
                    max=float(data.max()) if len(data) > 0 else None,
                    null_count=int(null_count),
                    null_percentage=float(null_percentage),
                    skewness=float(data.skew()) if len(data) > 0 else None,
                    kurtosis=float(data.kurtosis()) if len(data) > 0 else None,
                )
                
            elif col in categorical_features:
                data = df[col].dropna()
                value_counts = data.value_counts()
                
                stats[col] = FeatureStatistics(
                    feature_name=col,
                    data_type='categorical',
                    null_count=int(null_count),
                    null_percentage=float(null_percentage),
                    unique_values=int(data.nunique()),
                    most_common=str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    most_common_count=int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                )
            else:
                stats[col] = FeatureStatistics(
                    feature_name=col,
                    data_type='unknown',
                    null_count=int(null_count),
                    null_percentage=float(null_percentage),
                )
        
        return stats



class SeasonalityAnalyzer:
    
    SEASON_MAP = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
    }
    
    SEASON_ORDER = ['winter', 'spring', 'summer', 'autumn']
    
    EXPECTED_SEASONAL_DISTRIBUTION = {
        'winter': 0.25,
        'spring': 0.25,
        'summer': 0.25,
        'autumn': 0.25,
    }
    
    @staticmethod
    def calculate_gini_coefficient(distribution: Dict[str, int]) -> float:
        values = np.array(list(distribution.values()))
        if len(values) <= 1:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * np.sum(values)) - (n + 1) / n
        
        return float(np.clip(gini, 0, 1))
    
    @staticmethod
    def analyze_seasonal_distribution(df: pd.DataFrame) -> SeasonalityAnalysisResult:
        logger.info("Analyzing seasonal distribution")
        
        if 'T_season' not in df.columns:
            logger.warning("T_season column not found, analyzing T_month instead")
            if 'T_month' in df.columns:
                df_copy = df.copy()
                df_copy['T_season'] = df['T_month'].map(
                    lambda m: next(s for s, months in SeasonalityAnalyzer.SEASON_MAP.items() if m in months)
                    if pd.notna(m) else None
                )
            else:
                raise ValueError("Neither T_season nor T_month found in DataFrame")
        else:
            df_copy = df.copy()
        
        total_points = len(df_copy)
        seasonal_counts = {}
        seasonal_distributions = []
        
        for season in SeasonalityAnalyzer.SEASON_ORDER:
            count = (df_copy['T_season'] == season).sum()
            percentage = (count / total_points) * 100 if total_points > 0 else 0
            expected_percentage = SeasonalityAnalyzer.EXPECTED_SEASONAL_DISTRIBUTION[season] * 100
            
            tolerance = 3
            if abs(percentage - expected_percentage) <= tolerance:
                status = 'OK'
            elif percentage < expected_percentage - tolerance:
                status = 'LOW'
            else:
                status = 'HIGH'
            
            seasonal_counts[season] = count
            seasonal_distributions.append(SeasonalDistribution(
                season=season,
                total_points=count,
                percentage=round(percentage, 2),
                expected_percentage=round(expected_percentage, 2),
                status=status,
            ))
        
        gini = SeasonalityAnalyzer.calculate_gini_coefficient(seasonal_counts)
        
        if gini < 0.05:
            summary_status = 'IDEAL'
        elif gini < 0.15:
            summary_status = 'GOOD'
        else:
            summary_status = 'POOR'
        
        logger.info(f"[OK] Seasonal distribution analyzed:")
        for dist in seasonal_distributions:
            logger.info(f"   > {dist.season}: {dist.total_points} points ({dist.percentage}%), status: {dist.status}")
        logger.info(f"   > Gini coefficient: {gini:.4f} ({summary_status})")
        
        return SeasonalityAnalysisResult(
            total_points=total_points,
            seasonal_distributions=seasonal_distributions,
            gini_coefficient=gini,
            rare_combinations=[],
            summary_status=summary_status,
        )



class RareCombinationDetector:
    
    @staticmethod
    def detect_rare_combinations(df: pd.DataFrame,
                                 combination_factors: List[str],
                                 min_coverage: float = 0.02) -> List[RareCombination]:
        logger.info(f"Detecting rare combinations from {len(combination_factors)} factors...")
        
        available_factors = [f for f in combination_factors if f in df.columns]
        if len(available_factors) == 0:
            logger.warning(f"None of the combination factors found in DataFrame")
            return []
        
        if len(available_factors) < len(combination_factors):
            logger.warning(f"Not all factors available: {available_factors}")
        
        grouped = df.groupby(available_factors, observed=True).size().reset_index(name='count')
        
        total_combinations = len(grouped)
        total_points = len(df)
        
        n_unique_per_factor = [df[f].nunique() for f in available_factors]
        expected_combinations = np.prod(n_unique_per_factor) if len(n_unique_per_factor) > 0 else 1
        
        rare_combinations = []
        
        for _, row in grouped.iterrows():
            count = row['count']
            percentage = (count / total_points) * 100
            
            expected_count = total_points / expected_combinations if expected_combinations > 0 else total_points
            
            if percentage < min_coverage:
                combo_dict = {f: str(row[f]) for f in available_factors}
                combo_key = "_".join([f"{k}={v}" for k, v in combo_dict.items()])
                
                deficit = int(expected_count - count)
                if deficit > 100:
                    action = 'VAE'
                else:
                    action = 'SMOTE'
                
                rare_combinations.append(RareCombination(
                    combination_key=combo_key,
                    combination_dict=combo_dict,
                    current_points=int(count),
                    expected_points=int(expected_count),
                    deficit=deficit,
                    coverage_percentage=round(percentage / (100 / expected_combinations) * 100, 1),
                    action=action,
                ))
        
        rare_combinations.sort(key=lambda x: x.deficit, reverse=True)
        
        logger.info(f"[OK] Found {len(rare_combinations)} rare combinations out of {total_combinations} total")
        logger.info(f"   > Total points: {total_points}")
        logger.info(f"   > Expected combinations: {expected_combinations:.0f}")
        
        return rare_combinations



class FeatureStatisticsAnalyzer:
    
    def __init__(self):
        self.stats_calculator = FeatureStatisticsCalculator()
        self.seasonality_analyzer = SeasonalityAnalyzer()
        self.rare_detector = RareCombinationDetector()
        
        logger.info("[OK] FeatureStatisticsAnalyzer Запуск")
    
    def analyze(self, 
                df: pd.DataFrame,
                numerical_features: set,
                categorical_features: set,
                combination_factors: List[str] = None) -> Dict:
        logger.info(f"Starting feature statistics analysis for {len(df)} points...")
        
        results = {}
        
        logger.info("\nStep 1: Computing feature statistics")
        feature_stats = self.stats_calculator.calculate_statistics(
            df, numerical_features, categorical_features
        )
        results['feature_statistics'] = feature_stats
        logger.info(f"   [OK] Computed statistics for {len(feature_stats)} features")
        
        logger.info("\nStep 2: Analyzing seasonal distribution")
        seasonal_result = self.seasonality_analyzer.analyze_seasonal_distribution(df)
        results['seasonal_distribution'] = seasonal_result
        
        if combination_factors:
            logger.info("\nStep 3: Detecting rare combinations")
            rare_combos = self.rare_detector.detect_rare_combinations(
                df, combination_factors, min_coverage=0.02
            )
            results['rare_combinations'] = rare_combos
            logger.info(f"   [OK] Found {len(rare_combos)} rare combinations")
        
        logger.info("\n[DONE] Feature statistics analysis Готово!")
        
        return results
    
    def print_summary(self, results: Dict):
        logger.info("\n" + "=" * 80)
        logger.info("Feature statistics SUMMARY")
        logger.info("=" * 80)
        
        if 'seasonal_distribution' in results:
            sr = results['seasonal_distribution']
            logger.info(f"\nSeasonal Distribution (Total: {sr.total_points} points, Gini: {sr.gini_coefficient:.4f}):")
            for dist in sr.seasonal_distributions:
                status_icon = "[OK]" if dist.status == "IDEAL" else "[!]" if dist.status == "HIGH" else ">"
                logger.info(f"  {status_icon} {dist.season}: {dist.percentage}% (expected {dist.expected_percentage}%)")
        
        if 'rare_combinations' in results:
            rare_combos = results['rare_combinations']
            logger.info(f"\nRare Combinations (Top 10 by deficit):")
            for combo in rare_combos[:10]:
                logger.info(f"  - {combo.combination_key}")
                logger.info(f"    Current: {combo.current_points}, Expected: {combo.expected_points}, "
                           f"Deficit: {combo.deficit}, Action: {combo.action}")

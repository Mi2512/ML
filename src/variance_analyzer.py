
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import f_oneway, kruskal, levene
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ANOVAResult:
    feature_name: str
    f_statistic: float
    p_value: float
    is_significant: bool
    group_names: List[str]
    group_means: Dict[str, float]
    group_stds: Dict[str, float]
    effect_size: float


@dataclass
class VarianceAnalysisResult:
    anova_results: List[ANOVAResult]
    kruskal_results: List[Dict]
    levene_results: Dict
    significant_features: List[str]


class ANOVAAnalyzer:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, data: pd.DataFrame, feature_col: str, 
                group_col: str = 'T_season') -> Optional[VarianceAnalysisResult]:
        try:
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if group_col not in data.columns:
                self.logger.warning(f"Group column '{group_col}' not found")
                return None
            
            excluded = ['track_id', 'point_index', 'index', 'id', 'sample_id']
            features = [f for f in numerical_features if f not in excluded]
            
            anova_results = []
            kruskal_results = []
            levene_results = {}
            
            groups = data[group_col].unique()
            
            self.logger.info(f"\nAnalyzing {len(features)} features across {len(groups)} groups...")
            
            for feature in features:
                if data[feature].isna().sum() > len(data) * 0.5:
                    continue
                
                group_data = []
                group_means = {}
                group_stds = {}
                
                for group in groups:
                    group_vals = data[data[group_col] == group][feature].dropna().values
                    
                    if len(group_vals) > 1:
                        group_data.append(group_vals)
                        group_means[str(group)] = np.mean(group_vals)
                        group_stds[str(group)] = np.std(group_vals)
                
                if len(group_data) < 2:
                    continue
                
                try:
                    f_stat, p_value = f_oneway(*group_data)
                    
                    all_data = np.concatenate(group_data)
                    grand_mean = np.mean(all_data)
                    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
                    ss_total = sum((x - grand_mean) ** 2 for x in all_data)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    is_sig = p_value < 0.05
                    
                    result = ANOVAResult(
                        feature_name=feature,
                        f_statistic=f_stat,
                        p_value=p_value,
                        is_significant=is_sig,
                        group_names=[str(g) for g in groups],
                        group_means=group_means,
                        group_stds=group_stds,
                        effect_size=eta_squared
                    )
                    
                    anova_results.append(result)
                
                except Exception as e:
                    self.logger.debug(f"ANOVA error for {feature}: {e}")
                    continue
                
                try:
                    h_stat, p_kw = kruskal(*group_data)
                    kruskal_results.append({
                        'feature': feature,
                        'h_statistic': h_stat,
                        'p_value': p_kw,
                        'is_significant': p_kw < 0.05
                    })
                except:
                    pass
            
            try:
                for feature in features[:10]:
                    group_data = []
                    for group in groups:
                        group_vals = data[data[group_col] == group][feature].dropna().values
                        if len(group_vals) > 1:
                            group_data.append(group_vals)
                    
                    if len(group_data) > 1:
                        stat, p = levene(*group_data)
                        levene_results[feature] = {'statistic': stat, 'p_value': p}
            except:
                pass
            
            anova_results.sort(key=lambda x: x.p_value)
            
            significant = [r.feature_name for r in anova_results if r.is_significant]
            
            self.logger.info(f"[OK] ANOVA analysis complete")
            self.logger.info(f"     Significant features: {len(significant)}")
            
            if significant:
                self.logger.info(f"     Top features by F-statistic:")
                for i, result in enumerate(anova_results[:5]):
                    self.logger.info(
                        f"       {i+1}. {result.feature_name:30s}: F={result.f_statistic:8.4f}, p={result.p_value:.4f}, η²={result.effect_size:.4f}"
                    )
            
            return VarianceAnalysisResult(
                anova_results=anova_results,
                kruskal_results=kruskal_results,
                levene_results=levene_results,
                significant_features=significant
            )
        
        except Exception as e:
            self.logger.error(f"Variance analysis error: {e}")
            return None
    
    def export_to_dataframe(self, result: VarianceAnalysisResult) -> pd.DataFrame:
        if not result or not result.anova_results:
            return pd.DataFrame()
        
        data = {
            'feature': [r.feature_name for r in result.anova_results],
            'f_statistic': [r.f_statistic for r in result.anova_results],
            'p_value': [r.p_value for r in result.anova_results],
            'significant': [r.is_significant for r in result.anova_results],
            'effect_size_eta2': [r.effect_size for r in result.anova_results]
        }
        
        return pd.DataFrame(data).sort_values('f_statistic', ascending=False)


class VarianceAnalysisEngine:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer = ANOVAAnalyzer()
        self.result = None
    
    def analyze(self, data: pd.DataFrame, group_col: str = 'T_season') -> Optional[VarianceAnalysisResult]:
        self.logger.info("\n" + "="*80)
        self.logger.info("Variance analysis (ANOVA & KRUSKAL-WALLIS)")
        self.logger.info("="*80)
        
        self.result = self.analyzer.analyze(data, feature_col=None, group_col=group_col)
        
        return self.result
    
    def export_to_dataframe(self) -> pd.DataFrame:
        if self.result:
            return self.analyzer.export_to_dataframe(self.result)
        return pd.DataFrame()
    
    def get_significant_features(self) -> List[str]:
        if self.result:
            return self.result.significant_features
        return []

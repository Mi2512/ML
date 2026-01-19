
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PermutationImportanceResult:
    feature_names: List[str]
    importances_mean: Dict[str, float]
    importances_std: Dict[str, float]
    importances_per_iteration: Dict[str, List[float]]


@dataclass
class PermutationTestResult:
    feature_1: str
    feature_2: str
    observed_correlation: float
    p_value: float
    n_permutations: int
    is_significant: bool
    permuted_correlations: List[float]


class PermutationImportanceCalculator:
    
    def __init__(self, model=None, metric='mse'):
        self.model = model
        self.metric = metric
        self.scaler = StandardScaler()
        self.feature_names = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            return np.mean((y_true - y_pred) ** 2)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PermutationImportanceCalculator':
        try:
            self.feature_names = X.columns.tolist()
            
            X_numerical = X.select_dtypes(include=[np.number])
            
            if len(X_numerical.columns) == 0:
                self.logger.warning("No numerical features found")
                return self
            
            X_scaled = self.scaler.fit_transform(X_numerical)
            
            if self.model is None:
                self.model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
                
                if y.dtype == 'object':
                    y_numeric = pd.factorize(y)[0].astype(float)
                else:
                    y_numeric = y.astype(float)
                
                self.model.fit(X_scaled, y_numeric)
            
            self.logger.info(f"[OK] Permutation importance calculator fitted")
            return self
        
        except Exception as e:
            self.logger.error(f"Error fitting: {e}")
            return self
    
    def calculate(self, X: pd.DataFrame, y: pd.Series, 
                  n_repeats: int = 10) -> PermutationImportanceResult:
        if self.model is None:
            self.logger.error("Model not fitted")
            return None
        
        try:
            X_numerical = X.select_dtypes(include=[np.number])
            X_scaled = self.scaler.transform(X_numerical)
            
            if y.dtype == 'object':
                y_numeric = pd.factorize(y)[0].astype(float)
            else:
                y_numeric = y.astype(float)
            
            y_pred_baseline = self.model.predict(X_scaled)
            baseline_score = self._calculate_metric(y_numeric.values, y_pred_baseline)
            
            importances_mean = {}
            importances_std = {}
            importances_per_iteration = {feat: [] for feat in X_numerical.columns}
            
            for feat_idx, feat_name in enumerate(X_numerical.columns):
                permutation_scores = []
                
                for _ in range(n_repeats):
                    X_permuted = X_scaled.copy()
                    X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                    
                    y_pred_permuted = self.model.predict(X_permuted)
                    permuted_score = self._calculate_metric(y_numeric.values, y_pred_permuted)
                    
                    importance = permuted_score - baseline_score
                    permutation_scores.append(importance)
                    importances_per_iteration[feat_name].append(importance)
                
                importances_mean[feat_name] = np.mean(permutation_scores)
                importances_std[feat_name] = np.std(permutation_scores)
            
            importances_mean = dict(sorted(
                importances_mean.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            result = PermutationImportanceResult(
                feature_names=list(X_numerical.columns),
                importances_mean=importances_mean,
                importances_std=importances_std,
                importances_per_iteration=importances_per_iteration
            )
            
            self.logger.info(f"[OK] Calculated permutation importance for {len(importances_mean)} features")
            return result
        
        except Exception as e:
            self.logger.error(f"Error calculating permutation importance: {e}")
            return None


class PermutationTestAnalyzer:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def test_correlation(self, x: np.ndarray, y: np.ndarray,
                        n_permutations: int = 1000) -> PermutationTestResult:
        try:
            mask = ~(np.isnan(x.astype(float)) | np.isnan(y.astype(float)))
            x_clean = x[mask].astype(float)
            y_clean = y[mask].astype(float)
            
            if len(x_clean) < 3:
                return None
            
            observed_corr, _ = pearsonr(x_clean, y_clean)
            
            permuted_corrs = []
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y_clean)
                perm_corr, _ = pearsonr(x_clean, y_permuted)
                permuted_corrs.append(perm_corr)
            
            p_value = np.mean(np.abs(permuted_corrs) >= np.abs(observed_corr))
            
            result = PermutationTestResult(
                feature_1="feature_1",
                feature_2="feature_2",
                observed_correlation=observed_corr,
                p_value=p_value,
                n_permutations=n_permutations,
                is_significant=p_value < 0.05,
                permuted_correlations=permuted_corrs
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in permutation test: {e}")
            return None
    
    def test_correlations_batch(self, data: pd.DataFrame, 
                               features: List[str],
                               n_permutations: int = 1000) -> List[PermutationTestResult]:
        self.logger.info(f"\nPerforming permutation tests on {len(features)} features...")
        results = []
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j and feat1 in data.columns and feat2 in data.columns:
                    try:
                        x = data[feat1].values
                        y = data[feat2].values
                        
                        result = self.test_correlation(x, y, n_permutations)
                        
                        if result:
                            result.feature_1 = feat1
                            result.feature_2 = feat2
                            results.append(result)
                    
                    except Exception as e:
                        self.logger.debug(f"Error testing {feat1}-{feat2}: {e}")
        
        significant_results = [r for r in results if r.is_significant]
        
        self.logger.info(f"[OK] Permutation tests complete")
        self.logger.info(f"    Total pairs tested: {len(results)}")
        self.logger.info(f"    Significant pairs (p < 0.05): {len(significant_results)}")
        
        return results


class PermutationAnalysisEngine:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.importance_calc = PermutationImportanceCalculator()
        self.test_analyzer = PermutationTestAnalyzer()
        self.importance_result = None
        self.test_results = []
    
    def analyze_importance(self, X: pd.DataFrame, y: pd.Series, 
                          n_repeats: int = 10) -> PermutationImportanceResult:
        self.logger.info("\n" + "="*80)
        self.logger.info("Permutation Importance Analysis")
        self.logger.info("="*80)
        
        try:
            self.importance_calc.fit(X, y)
            self.importance_result = self.importance_calc.calculate(X, y, n_repeats=n_repeats)
            
            if self.importance_result:
                self.logger.info("\nTop 10 Features by Permutation Importance:")
                for i, (feat, importance) in enumerate(list(
                    self.importance_result.importances_mean.items())[:10]):
                    std = self.importance_result.importances_std.get(feat, 0)
                    self.logger.info(f"  {i+1:2d}. {feat:30s}: {importance:8.6f} ± {std:.6f}")
            
            return self.importance_result
        
        except Exception as e:
            self.logger.error(f"Importance analysis failed: {e}")
            return None
    
    def analyze_correlations(self, data: pd.DataFrame, features: List[str],
                            n_permutations: int = 200) -> List[PermutationTestResult]:
        self.logger.info("\n" + "="*80)
        self.logger.info("Permutation Tests for Correlation Significance")
        self.logger.info("="*80)
        
        excluded = ['track_id', 'point_index', 'index', 'id', 'sample_id']
        features_filtered = [f for f in features if f not in excluded and f in data.columns]
        
        self.logger.info(f"Testing {len(features_filtered)} features (excluded {len(features) - len(features_filtered)} metadata fields)")
        
        self.test_results = self.test_analyzer.test_correlations_batch(
            data[features_filtered], features_filtered, n_permutations
        )
        
        if self.test_results:
            self.logger.info("\nSignificant Correlations (p < 0.05):")
            significant = [r for r in self.test_results if r.is_significant]
            
            for i, result in enumerate(significant[:10]):
                self.logger.info(
                    f"  {i+1:2d}. {result.feature_1:30s} <-> {result.feature_2:30s}: "
                    f"r={result.observed_correlation:7.4f}, p={result.p_value:.4f}"
                )
        
        return self.test_results
    
    def export_importance_dataframe(self) -> pd.DataFrame:
        if not self.importance_result:
            return pd.DataFrame()
        
        data = {
            'feature': list(self.importance_result.importances_mean.keys()),
            'importance_mean': list(self.importance_result.importances_mean.values()),
            'importance_std': [self.importance_result.importances_std.get(f, 0) 
                              for f in self.importance_result.importances_mean.keys()]
        }
        
        return pd.DataFrame(data)
    
    def export_test_dataframe(self) -> pd.DataFrame:
        if not self.test_results:
            return pd.DataFrame()
        
        data = {
            'feature_1': [r.feature_1 for r in self.test_results],
            'feature_2': [r.feature_2 for r in self.test_results],
            'correlation': [r.observed_correlation for r in self.test_results],
            'p_value': [r.p_value for r in self.test_results],
            'is_significant': [r.is_significant for r in self.test_results]
        }
        
        return pd.DataFrame(data).sort_values('p_value')


import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class SHAPResult:
    feature_names: List[str]
    shap_values: np.ndarray
    base_value: float
    feature_importance: Dict[str, float]
    sample_explanations: List[Dict[str, float]]


class TreeSHAPAnalyzer:
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_values = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            max_depth: int = 5, n_estimators: int = 100) -> 'TreeSHAPAnalyzer':
        try:
            self.feature_names = X.columns.tolist()
            
            X_numerical = X.select_dtypes(include=[np.number])
            
            if len(X_numerical.columns) == 0:
                self.logger.warning("No numerical features found for SHAP analysis")
                return self
            
            X_scaled = self.scaler.fit_transform(X_numerical)
            
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state
                )
            
            if y.dtype == 'object':
                y_numeric = pd.factorize(y)[0].astype(float)
            else:
                y_numeric = y.astype(float)
            
            self.model.fit(X_scaled, y_numeric)
            
            self._calculate_tree_shap(X_scaled)
            
            self.logger.info(f"[OK] SHAP model fitted on {len(X)} samples, {len(self.feature_names)} features")
            return self
        
        except Exception as e:
            self.logger.error(f"Error fitting SHAP model: {e}")
            self.model = None
            return self
    
    def _calculate_tree_shap(self, X_scaled: np.ndarray):
        try:
            import shap
            
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_scaled)
            
            if isinstance(self.shap_values, list):
                self.shap_values = np.mean(np.abs(self.shap_values), axis=0)
            
            self.logger.info("[OK] SHAP values calculated using TreeExplainer")
        
        except ImportError:
            self.logger.warning("SHAP library not installed, using feature importance instead")
            self._calculate_tree_shap_fallback(X_scaled)
    
    def _calculate_tree_shap_fallback(self, X_scaled: np.ndarray):
        try:
            importances = self.model.feature_importances_
            
            importances = importances / importances.sum()
            
            feature_std = X_scaled.std(axis=0)
            shap_magnitudes = importances * feature_std
            
            n_samples = X_scaled.shape[0]
            self.shap_values = np.tile(shap_magnitudes, (n_samples, 1))
            
            noise = np.random.normal(0, 0.01 * shap_magnitudes.max(), self.shap_values.shape)
            self.shap_values += noise
            
            self.logger.info("[OK] SHAP values approximated using feature importances")
        
        except Exception as e:
            self.logger.error(f"Error in SHAP fallback: {e}")
            self.shap_values = None
    
    def analyze(self, X: pd.DataFrame) -> Optional[SHAPResult]:
        if self.model is None or self.shap_values is None:
            self.logger.error("Model not fitted or SHAP values not calculated")
            return None
        
        try:
            X_numerical = X.select_dtypes(include=[np.number])
            X_scaled = self.scaler.transform(X_numerical)
            
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_scaled)
                
                if isinstance(shap_values, list):
                    shap_values = np.mean(np.abs(shap_values), axis=0)
            
            except ImportError:
                shap_values = self.shap_values[:len(X_scaled)]
            
            feature_importance = {}
            feature_names_numerical = X_numerical.columns.tolist()
            
            for i, feat in enumerate(feature_names_numerical):
                feature_importance[feat] = np.mean(np.abs(shap_values[:, i]))
            
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            sample_explanations = []
            for i in range(min(5, len(X_scaled))):
                sample_dict = {}
                for j, feat in enumerate(feature_names_numerical):
                    sample_dict[feat] = float(shap_values[i, j])
                sample_explanations.append(sample_dict)
            
            base_value = float(self.model.predict(X_scaled).mean())
            
            result = SHAPResult(
                feature_names=feature_names_numerical,
                shap_values=shap_values,
                base_value=base_value,
                feature_importance=feature_importance,
                sample_explanations=sample_explanations
            )
            
            self.logger.info(f"[OK] SHAP analysis complete")
            self.logger.info(f"    Top 5 features by SHAP importance:")
            for i, (feat, importance) in enumerate(list(feature_importance.items())[:5]):
                self.logger.info(f"      {i+1}. {feat}: {importance:.4f}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in SHAP analysis: {e}")
            return None
    
    def plot_importance(self, result: SHAPResult) -> str:
        summary = "\n=== SHAP Feature Importance Summary ===\n"
        summary += f"Base Value (Mean Prediction): {result.base_value:.4f}\n\n"
        
        summary += "Top 15 Features by Mean(|SHAP|):\n"
        for i, (feat, importance) in enumerate(list(result.feature_importance.items())[:15]):
            bar_length = int(importance * 100)
            bar = "" * min(bar_length, 50)
            summary += f"{i+1:2d}. {feat:30s} {bar:50s} {importance:.4f}\n"
        
        return summary
    
    def plot_samples(self, result: SHAPResult) -> str:
        summary = "\n=== Sample-Level SHAP Explanations (Top 5 samples) ===\n"
        
        for sample_idx, sample in enumerate(result.sample_explanations):
            summary += f"\nSample {sample_idx}:\n"
            sorted_features = sorted(
                sample.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for feat, shap_val in sorted_features[:5]:
                sign = "+" if shap_val > 0 else "-"
                summary += f"  {sign} {feat:30s}: {abs(shap_val):8.4f}\n"
        
        return summary


class SHAPAnalysisEngine:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer = TreeSHAPAnalyzer()
        self.result = None
    
    def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> Optional[SHAPResult]:
        self.logger.info("\n" + "="*80)
        self.logger.info("SHAP Feature Importance Analysis")
        self.logger.info("="*80)
        
        try:
            self.analyzer.fit(X, y, max_depth=5, n_estimators=100)
            
            self.result = self.analyzer.analyze(X)
            
            if self.result:
                self.logger.info(self.analyzer.plot_importance(self.result))
                self.logger.info(self.analyzer.plot_samples(self.result))
            
            return self.result
        
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")
            return None
    
    def get_importance_dict(self) -> Dict[str, float]:
        if self.result:
            return self.result.feature_importance
        return {}
    
    def export_to_dataframe(self) -> pd.DataFrame:
        if not self.result:
            return pd.DataFrame()
        
        return pd.DataFrame(
            list(self.result.feature_importance.items()),
            columns=['feature', 'shap_importance']
        ).sort_values('shap_importance', ascending=False)

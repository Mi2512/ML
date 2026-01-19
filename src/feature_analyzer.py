
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    
    def __init__(self):
        logger.info("FeatureAnalyzer Запуск")
        try:
            from feature_correlation_analyzer import FeatureCorrelationAnalyzer
            self.correlator = FeatureCorrelationAnalyzer()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not import correlator: {e}")
            self.correlator = None
    
    def analyze_correlations(self, features_df) -> Dict:
        if self.correlator:
            return self.correlator.analyze_all_methods(features_df)
        
        return {"status": "error", "message": "Correlator unavailable"}
    
    def compute_pearson_correlation(self, features_df) -> Dict:
        if self.correlator:
            return self.correlator.compute_pearson(features_df)
        
        return {}
    
    def compute_spearman_correlation(self, features_df) -> Dict:
        if self.correlator:
            return self.correlator.compute_spearman(features_df)
        
        return {}
    
    def compute_cramers_v(self, features_df) -> Dict:
        if self.correlator:
            return self.correlator.compute_cramers_v(features_df)
        
        return {}
    
    def compute_mutual_information(self, features_df) -> Dict:
        if self.correlator:
            return self.correlator.compute_mutual_information(features_df)
        
        return {}
    
    def detect_confounders(self, features_df, target: str) -> List[str]:
        if self.correlator:
            result = self.correlator.detect_confounders(features_df, target)
            if isinstance(result, dict):
                return result.get("confounders", [])
            return result
        
        return []
    
    def identify_redundant_features(self, features_df, threshold: float = 0.95) -> List[str]:
        if self.correlator:
            result = self.correlator.identify_redundant_features(features_df, threshold)
            if isinstance(result, dict):
                return result.get("redundant_features", [])
            return result
        
        return []


CorrelationAnalyzer = FeatureAnalyzer
